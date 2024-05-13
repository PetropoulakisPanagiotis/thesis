#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <g2o/types/sba/types_sba.h>

#include "python/core/base_vertex.h"
#include "python/core/base_binary_edge.h"


namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

void declareTypesSBA(py::module & m) {

    py::class_<VertexIntrinsics, BaseVertex<4, Eigen::Matrix<double, 5, 1, Eigen::ColMajor>> >(m, "VertexIntrinsics")
        .def(py::init<>())
        .def("set_to_origin_impl", &VertexIntrinsics::setToOriginImpl)
        .def("oplus_impl", &VertexIntrinsics::oplusImpl)    // double* -> void
    ;


    templatedBaseVertex<6, SBACam>(m, "_6_SBACam");
    py::class_<VertexCam, BaseVertex<6, SBACam>>(m, "VertexCam")
        .def(py::init<>())
        .def("set_to_origin_impl", &VertexCam::setToOriginImpl)
        .def("set_estimate", &VertexCam::setEstimate)   // const SBACam& -> void
        .def("oplus_impl", &VertexCam::oplusImpl)
        .def("set_estimate_data_impl", &VertexCam::setEstimateDataImpl)
        .def("get_estimate_data", &VertexCam::getEstimateData)
        .def("estimate_dimension", &VertexCam::estimateDimension)
        .def("set_minimal_estimate_data_impl", &VertexCam::setMinimalEstimateDataImpl)
        .def("get_minimal_estimate_data", &VertexCam::getMinimalEstimateData)
        .def("minimal_estimate_dimension", &VertexCam::minimalEstimateDimension)
    ;

    templatedBaseVertex<6, CustomCam>(m, "_6_CustomCam");
    py::class_<VertexCustomCam, BaseVertex<6, CustomCam>>(m, "VertexCustomCam")
        .def(py::init<>())
        .def("set_to_origin_impl", &VertexCustomCam::setToOriginImpl)
        .def("set_estimate", &VertexCustomCam::setEstimate)   // const SBACam& -> void
        .def("oplus_impl", &VertexCustomCam::oplusImpl)
        .def("set_estimate_data_impl", &VertexCustomCam::setEstimateDataImpl)
        .def("get_estimate_data", &VertexCustomCam::getEstimateData)
        .def("estimate_dimension", &VertexCustomCam::estimateDimension)
        .def("set_minimal_estimate_data_impl", &VertexCustomCam::setMinimalEstimateDataImpl)
        .def("get_minimal_estimate_data", &VertexCustomCam::getMinimalEstimateData)
        .def("minimal_estimate_dimension", &VertexCustomCam::minimalEstimateDimension)
    ;

    /* Added */
    py::class_<VertexCustomXYZ, BaseVertex<3, Vector3D>>(m, "VertexCustomXYZ")
        .def(py::init<>())
        .def("set_to_origin_impl", &VertexCustomXYZ::setToOriginImpl)
        .def("oplus_impl", &VertexCustomXYZ::oplusImpl)    // double* -> void
    ;

    
    /* Added */
    py::class_<VertexCanonical, BaseVertex<1, double>>(m, "VertexCanonical")
        .def(py::init<>())
        .def("set_to_origin_impl", &VertexCanonical::setToOriginImpl)
        .def("oplus_impl", &VertexCanonical::oplusImpl)    // double* -> void
    ;

    /* Added */
    py::class_<VertexScale, BaseVertex<1, double>>(m, "VertexScale")
        .def(py::init<>())
        .def("set_to_origin_impl", &VertexScale::setToOriginImpl)
        .def("oplus_impl", &VertexScale::oplusImpl)    // double* -> void
    ;




    py::class_<VertexSBAPointXYZ, BaseVertex<3, Vector3D>>(m, "VertexSBAPointXYZ")
        .def(py::init<>())
        .def("set_to_origin_impl", &VertexSBAPointXYZ::setToOriginImpl)
        .def("oplus_impl", &VertexSBAPointXYZ::oplusImpl)    // double* -> void
    ;

    // monocular projection
    // first two args are the measurement type, second two the connection classes
    templatedBaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexCam>(m, "_2_Vector2D_VertexSBAPointXYZ_VertexCam");
    py::class_<EdgeProjectP2MC, BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexCam>>(m, "EdgeProjectP2MC")
        .def(py::init<>())
        .def("compute_error", &EdgeProjectP2MC::computeError)    // () -> void
        .def("linearize_oplus", &EdgeProjectP2MC::linearizeOplus)
    ;



    // added stereo projection
    // first two args are the measurement type, second two the connection classes
    templatedBaseBinaryEdge<3, Vector3D, VertexCustomXYZ, VertexCustomCam>(m, "_3_Vector3D_VertexCustomXYZ_VertexCustomCam");
    py::class_<EdgeStereo, BaseBinaryEdge<3, Vector3D, VertexCustomXYZ, VertexCustomCam>>(m, "EdgeStereo")
        .def(py::init<>())
        .def("compute_error", &EdgeStereo::computeError)    // () -> void
        .def("linearize_oplus", &EdgeStereo::linearizeOplus)
    ;



    templatedBaseUnaryEdge<1, double, VertexScale>(m, "_1_double_VertexScale");
    py::class_<EdgeScaleNetworkConsistency, BaseUnaryEdge<1, double, VertexScale>>(m, "EdgeScaleNetworkConsistency")
        .def(py::init<>())
        .def("compute_error", &EdgeScaleNetworkConsistency::computeError)    // () -> void
        .def("linearize_oplus", &EdgeScaleNetworkConsistency::linearizeOplus)
    ;



    /*  
    // Depth consistency scale
    py::class_<EdgeDepthConsistencyScale, BaseMultiEdge<1, double>>(m, "EdgeDepthConsistencyScale")
        .def(py::init<>())
        .def("compute_error", &EdgeDepthConsistencyScale::computeError)    // () -> void
        .def("linearize_oplus", &EdgeDepthConsistencyScale::linearizeOplus)
    ;
    */
   
    templatedBaseMultiEdge<1, double>(m, "_1_double");
    py::class_<EdgeDepthConsistencyScale, BaseMultiEdge<1, double>>(m, "EdgeDepthConsistencyScale")
        .def(py::init<>())
        .def("compute_error", &EdgeDepthConsistencyScale::computeError)    // () -> void
    ;




    // stereo projection
    // first two args are the measurement type, second two the connection classes
    templatedBaseBinaryEdge<3, Vector3D, VertexSBAPointXYZ, VertexCam>(m, "_3_Vector3D_VertexSBAPointXYZ_VertexCam");
    py::class_<EdgeProjectP2SC, BaseBinaryEdge<3, Vector3D, VertexSBAPointXYZ, VertexCam>>(m, "EdgeProjectP2SC")
        .def(py::init<>())
        .def("compute_error", &EdgeProjectP2SC::computeError)    // () -> void
        .def("linearize_oplus", &EdgeProjectP2SC::linearizeOplus)
    ;


    // monocular projection with parameter calibration
    // first arg are the measurement type, second the connection classes
    py::class_<EdgeProjectP2MC_Intrinsics, BaseMultiEdge<2, Vector2D>>(m, "EdgeProjectP2MC_Intrinsics")
        .def(py::init<>())
        .def("compute_error", &EdgeProjectP2MC_Intrinsics::computeError)    // () -> void
        .def("linearize_oplus", &EdgeProjectP2MC_Intrinsics::linearizeOplus)
    ;


    templatedBaseBinaryEdge<6, SE3Quat, VertexCam, VertexCam>(m, "_6_SE3Quat_VertexCam_VertexCam");
    py::class_<EdgeSBACam, BaseBinaryEdge<6, SE3Quat, VertexCam, VertexCam>>(m, "EdgeSBACam")
        .def(py::init<>())
        .def("compute_error", &EdgeSBACam::computeError)  
        .def("set_measurement", &EdgeSBACam::setMeasurement)
        .def("initial_estimate_possible", &EdgeSBACam::initialEstimatePossible)
        .def("set_measurement_data", &EdgeSBACam::setMeasurementData)
        .def("get_measurement_data", &EdgeSBACam::getMeasurementData)
        .def("measurement_dimension", &EdgeSBACam::measurementDimension)
        .def("set_measurement_from_state", &EdgeSBACam::setMeasurementFromState)
    ;


    templatedBaseBinaryEdge<1, double, VertexCam, VertexCam>(m, "_1_double_VertexCam_VertexCam");
    py::class_<EdgeSBAScale, BaseBinaryEdge<1, double, VertexCam, VertexCam>>(m, "EdgeSBAScale")
        .def(py::init<>())
        .def("compute_error", &EdgeSBAScale::computeError) 
        .def("set_measurement", &EdgeSBAScale::setMeasurement)
        .def("initial_estimate_possible", &EdgeSBAScale::initialEstimatePossible)
        .def("initial_estimate", &EdgeSBAScale::initialEstimate)
    ;

}

}  // end namespace g2o
