#include <tuple>

#include <ATen/ATen.h>

#include "utils.h"

/***********************************************************************************************************************
 * Utility functions
 **********************************************************************************************************************/

at::Tensor normalize_shape(const at::Tensor& x) {
  if (x.ndimension() == 1) {
    return x.reshape({1, -1, 1});
  } else {
    return x.reshape({x.size(0), x.size(1), -1});
  }
}
