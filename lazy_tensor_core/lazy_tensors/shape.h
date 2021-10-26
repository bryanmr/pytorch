#pragma once

#include <atomic>
#include <ostream>
#include <vector>

#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensors/computation_client/client_data.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/layout.h"
#include "lazy_tensors/span.h"
#include "lazy_tensors/str_cat.h"
#include "lazy_tensors/str_join.h"
#include "lazy_tensors/types.h"

namespace lazy_tensors {

class Shape {
 public:
  Shape() : at_element_type_(c10::ScalarType::Undefined) {}

  Shape(at::ScalarType element_type,
        lazy_tensors::Span<const int64> dimensions);

  // Shape(PrimitiveType element_type, lazy_tensors::Span<const int64>
  // dimensions)
  //     : element_type_(element_type),
  //       dimensions_(dimensions.begin(), dimensions.end()),
  //       dynamic_dimensions_(dimensions.size(), false) {}

  Shape(lazy_tensors::Span<const Shape> element_shapes)
      : is_tuple_(true),
        at_element_type_(c10::ScalarType::Undefined),
        element_shapes_(element_shapes.begin(), element_shapes.end()) {}

  Shape(const client::ShapeData& shape_data);

  std::string ToString(bool print_layout = false) const {
    return lazy_tensors::StrCat(toString(at_element_type_), "[",
                                c10::Join(",", dimensions_), "]");
  }

  int64 rank() const { return dimensions_.size(); }

  bool IsArray() const { return false; }

  bool IsTuple() const { return is_tuple_; }

  bool is_dynamic_dimension(int dimension) const {
    return dynamic_dimensions_.at(dimension);
  }

  void set_dynamic_dimension(int dimension, bool is_dynamic) {
    dynamic_dimensions_[dimension] = is_dynamic;
  }

  lazy_tensors::Span<const bool> dynamic_dimensions() const {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  // Removes the dimension at index dim_to_delete entirely, reducing the rank
  // by 1.
  void DeleteDimension(int64 dim_to_delete);

  c10::ScalarType at_element_type() const { return at_element_type_; }
  void set_element_type(at::ScalarType value);

  // Methods for accessing the dimensions array.
  int dimensions_size() const { return dimensions_.size(); }
  int64 dimensions(int index) const {
    if (dynamic_mode_.load()) {
      throw std::runtime_error("Exact shape not known");
    }
    LTC_CHECK_LT(index, dimensions_.size());
    return dimensions_[index];
  }

  void set_dimensions(int index, int64 value) {
    LTC_CHECK_LT(index, dimensions_.size());
    dimensions_[index] = value;
  }

  lazy_tensors::Span<const int64> dimensions() const {
    if (dynamic_mode_.load()) {
      throw std::runtime_error("Exact shape not known");
    }
    return MakeSpan(dimensions_);
  }

  int tuple_shapes_size() const { return element_shapes_.size(); }

  const Shape& tuple_shapes(int index) const {
    LTC_CHECK_GE(index, 0);
    LTC_CHECK_LT(index, element_shapes_.size());
    return element_shapes_[index];
  }
  const std::vector<Shape>& tuple_shapes() const { return element_shapes_; }

  const Layout& layout() const { return layout_; }

  Layout* mutable_layout() { return &layout_; }

  bool operator==(const Shape& other) const {
    return at_element_type_ == other.at_element_type_ &&
           dimensions_ == other.dimensions_;
  }

  static bool IsDynamicMode();

  static void SetDynamicMode();

 private:
  bool is_tuple_ = false;
  c10::ScalarType at_element_type_;
  std::vector<int64> dimensions_;
  std::vector<bool> dynamic_dimensions_;
  std::vector<Shape> element_shapes_;
  Layout layout_;
  static std::atomic<bool> dynamic_mode_;
};

class ProgramShape {
 public:
  ProgramShape(std::vector<Shape> parameters,
               std::vector<std::string> parameter_names, Shape result)
      : parameters_(std::move(parameters)),
        parameter_names_(std::move(parameter_names)),
        result_(std::move(result)) {
    LTC_CHECK_EQ(parameters_.size(), parameter_names_.size());
  }

  int parameters_size() const { return parameters_.size(); }

  const std::vector<Shape>& parameters() const { return parameters_; }

  const std::vector<std::string>& parameter_names() const {
    return parameter_names_;
  }

  const Shape& result() const { return result_; }

 private:
  std::vector<Shape> parameters_;
  std::vector<std::string> parameter_names_;
  Shape result_;
};

inline std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  return out << shape.ToString();
}

// TODO(whc) took away inline temporarily, figured we'd delete this anyway
client::ShapeData ToShapeData(const Shape& shape);

}  // namespace lazy_tensors
