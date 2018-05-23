def _flops_cell(arc, incs, outc, w, h, num_cells=5):
  """Compute the number of ops in enas layer.
  Batch norm and relu not counted.

  There are 5 op types:
  0: sep3x3 1
  1: sep5x5 2
  2: avg3x3 1
  3: max3x3 1
  4: id 0
  5: bottleneck 5
  6: inverted bn 20
  """
  def _sep(op, inc, outc, w, h):
    if op == 0:
      return (9 + outc) * inc * w * h
    if op == 1:
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
    if op in [0, 1]:
      flops += _sep(op, inc, outc, w, h)
    elif op in [2, 3]:
      flops += _pool(inc, outc, w, h)
    else:
      flops += _id(inc, outc, w, h)
    c = (num_cells + 2 - len(used)) * outc
  return flops, c


def count_flops(normal_arc, reduce_arc, num_layers):
  """Count FLOPs of enas model.
  Reductions and calibrations are ignored."""
  w, h = (4, 4)
  total_flops = 0
  pool_distance = num_layers // 3
  pool_layers = [pool_distance, 2 * pool_distance + 1]
  # first input has width 3 * out_filters
  incs = [3, 3]
  outc = 1
  for layer in range(num_layers + 2):
    if layer not in pool_layers:
      flops, c += _flops_cell(normal_arc, incs, outc, w, h)
      total_flops += flops
    else:
      # skipping reduction and calibration
      outc *= 2
      incs = [outc, outc]
      w /= 2
      h /= 2
      flops, c += _flops_cell(reduce_arc, incs, outc, w, h)
      total_flops += flops
    incs = [incs[-1], c]
  return total_flops



