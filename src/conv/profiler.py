def _flops_cell(arc, incs, outc, w, h, num_cells=5):
  """Compute the number of ops in enas layer.
  Batch norm and relu not counted.

  There are 5 op types:
  0: sep3x3 1
  1: sep3x3 1
  2: sep5x5 2
  3: avg3x3 1
  4: max3x3 1
  5: id 0
  6: empty
  7: inverted bn 20 not implemented
  """
  def _sep(op, inc, outc, w, h):
    if op < 2:
      return (9 + outc) * inc * w * h
    else:
      return (25 + outc) * inc * w * h

  def _pool(inc, outc, w, h):
    if inc == outc:
      return 9 * inc * w * h
    else:
      return (9 * inc + inc * outc) * w * h

  def _id(inc, outc, w, h):
    if inc == outc:
      return 0
    else:
      return inc * outc * w * h

  flops = 0
  used = [0, 1]
  for cell_id in range(num_cells * 2):
    ind = arc[2 * cell_id]
    if ind not in used:
      used.append(ind)
    if ind < 2:
      inc = incs[ind]
    else:
      inc = outc
    op = arc[2 * cell_id + 1]
    if op in [0, 1, 2]:
      flops += _sep(op, inc, outc, w, h)
    elif op in [3, 4]:
      flops += _pool(inc, outc, w, h)
    elif op in [5]:
      flops += _id(inc, outc, w, h)
    else:
      # check dropped
      if cell_id % 2 == 1 and arc[2 * cell_id - 1] == 6:
        if cell_id // 2 not in used:
          used.append(cell_id // 2)
  c = (num_cells + 2 - len(used)) * outc
  return flops, c


def _calibration(inc, outc, w, h):
  return inc * outc * w * h


def count_flops(normal_arc, reduce_arc, num_layers):
  """Count FLOPs of enas model.
  Reductions and calibrations are ignored."""
  w, h = (4, 4)
  total_flops = 0
  pool_distance = num_layers // 3
  pool_layers = [pool_distance, 2 * pool_distance + 1]
  # first input has width 3 * out_filters
  outc = 1
  # apply calibration outside enas layer to save computations
  # total_flops += _calibration(3, outc, w, h)
  incs = [outc, outc]
  for layer in range(num_layers + 2):
    if layer not in pool_layers:
      flops, c = _flops_cell(normal_arc, incs, outc, w, h)
      total_flops += flops
    else:
      outc *= 2
      w /= 2
      h /= 2
      total_flops += 2 * _calibration(incs[0], outc, w, h)
      incs = [outc, outc]
      flops, c = _flops_cell(reduce_arc, incs, outc, w, h)
      total_flops += flops
    # apply calibration following enas output to save computations
    total_flops += _calibration(c, outc, w, h)
    incs = [incs[-1], outc]
  return total_flops


if __name__ == '__main__':
  import numpy as np
  normal_arc = np.array([1, 6, 0, 1, 1, 0, 0, 6, 1, 1, 0, 4, 0, 0, 4, 1, 0, 4, 1, 4])
  reduce_arc = np.array([0, 5, 1, 1, 1, 6, 0, 2, 0, 5, 0, 3, 0, 0, 0, 5, 2, 3, 0, 1])
  # normal_arc = [1, 1, 0, 4, 1, 4, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 2, 1, 0]
  # reduce_arc = [1, 5, 0, 1, 1, 2, 0, 4, 3, 1, 0, 3, 1, 1, 0, 2, 1, 1, 0, 2]
  print('FLOPS factor: %.1fK' % (count_flops(
    normal_arc, reduce_arc, 6) / 1000.0))
