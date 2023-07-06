//
// Created by user on 2023/2/27.
//

#ifndef MAINLOOP_DYNAMICOBJECTMANAGER_H
#define MAINLOOP_DYNAMICOBJECTMANAGER_H
#include "Eigen/Core"
//#include "sophus/se3.hpp"

#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/solver.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/stuff/sampler.h"
#include "g2o/types/icp/types_icp.h"
//class ObjPose : public g2o::BaseVertex<6, Sophus::SE3d>
//{
//public:
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
//
//    virtual void setToOriginImpl() override{
//        _estimate = Sophus::SE3d();
//    }
//
//    virtual void oplusImpl(const double* update) override{
//        Eigen::Matrix<double, 6, 1> update_eigen;
//        //std::cout<<"update:"<<update[0]<<","<<update[1]<<","<<update[2]<<","<<update[3]<<","<<update[4]<<","<<update[5]<<std::endl;
//        update_eigen << update[0], update[1], update[2],
//                update[3], update[4], update[5];
//        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
//    }
//
//    virtual bool read(std::istream& in) override{}
//    virtual bool write(std::ostream& out) const override{}
//};
//
//class EdgeFlowDepth : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, ObjPose>
//{
//public:
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
//    EdgeFlowDepth(Eigen::Vector3d vertex): _vertex(vertex)
//    {
//        //resize(VERTEX_KNN); //only for BaseVariableSizedEdge
//    }
//    bool read(std::istream& is) { return 0; }
//    bool write(std::ostream& os)const { return 0; }
//
//    virtual void computeError() override
//    {
//        const ObjPose* v = static_cast<ObjPose*>(_vertices[0]);
//        Sophus::SE3d T = v->estimate();
//        Eigen::Vector3d predict = T * _vertex;
//        _error = Eigen::Vector3d(predict - _measurement);
//    }
//
//
//private:
//    Eigen::Vector3d _vertex;
//};
class DynamicObjectManager{
public:
    DynamicObjectManager()
    {
        std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
        std::unique_ptr <g2o::BlockSolverX> solver_ptr ( new g2o::BlockSolverX( std::move(linearSolver)) );
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
        optimizer_.setAlgorithm(solver);
        optimizer_.setVerbose(true);
    }

    void trackObjects(const uchar4* imgSrc, const uchar4* imgTar, const float * depSrc, const float* depTar,
                      const float* flowU, const float* flowV, Eigen::Matrix4f intrinsics, uint currFrameNumber,
                      Eigen::Matrix4f prev2world, Eigen::Matrix4f curr2world, uchar* personMask, uchar* instanceMask,
                      uint width, uint height)
    {
//        ofstream ofs;
//        ofs.open("tracking/"+std::to_string(currFrameNumber)+".txt",ios::out );
//        for (int i = 0;i<g_objectCloud.size();i++) {
//            Eigen::Vector4f worldc = g_objPoses[g_objPoses.size() - 1] * g_objectCloud[i];
//            if (!std::isinf(worldc.x())&&!std::isinf(worldc.y())&&!std::isinf(worldc.z()))
//                ofs << worldc.x() << "," << worldc.y() << "," << worldc.z() << endl;
//        }
//        ofs.close();

        cv::Mat optimizedMat_src = ucharToMat(imgSrc, width, height);
        cv::Mat optimizedMat_tar = ucharToMat(imgTar, width, height);

        optimizer_.clear();

        // set up two poses
        int vertex_id = 0;
        for (size_t i = 0; i < 2; ++i) {
            // set up rotation and translation for this node
            Eigen::Vector3d t(0, 0, 0);
            Eigen::Quaterniond q;
            q.setIdentity();

            Eigen::Isometry3d cam;  // camera pose
            cam = q;
            cam.translation() = t;

            // set up node
            g2o::VertexSE3* vc = new g2o::VertexSE3();
            vc->setEstimate(cam);

            vc->setId(vertex_id);  // vertex id

            cerr << t.transpose() << " | " << q.coeffs().transpose() << endl;

            // set first cam pose fixed
            if (i == 0) vc->setFixed(true);

            // add to optimizer
            optimizer_.addVertex(vc);

            vertex_id++;
        }

//        ObjPose * pose =new ObjPose();
//        pose->setId(0);
//        pose->setEstimate(Sophus::SE3d());
//        optimizer_.addVertex(pose);
        bool hasTrackingLabel = false;
        for(int i = 0;i<width * height;i++)
        {
            if(instanceMask[i] == g_objectCat)
            {
                hasTrackingLabel = true;
                break;
            }
        }
        std::cout<<"g_objectCloud: "<<g_objectCloud.size()<<std::endl;
        std::cout<<"tracking frame:"<<currFrameNumber<<std::endl;
        for(int i = 0;i<g_objectCloud.size();i++)
        {
            Eigen::Vector4f last_state = g_objPoses[g_objPoses.size() - 1] * g_objectCloud[i];
            //flow
            //Eigen::Vector4f last_prev = prev2world.inverse() * last_state;


            float2 screenSrc = make_float2(
                    last_state.x() * intrinsics(0, 0) / last_state.z() + intrinsics(0, 2),
                    last_state.y() * intrinsics(1, 1) / last_state.z() + intrinsics(1, 2));
            cv::Point2f cvP(screenSrc.x, screenSrc.y);
            cv::circle(optimizedMat_src, cvP, 3, cv::Scalar(255, 0, 120), -1);
            float dSrc = bilinearInterpolationFloat(screenSrc.x, screenSrc.y, depSrc, width, height);
            Eigen::Vector4f dotSrc;
            dotSrc = Eigen::Vector4f(dSrc * (screenSrc.x - intrinsics(0,2)) / intrinsics(0,0),
                                     dSrc * (screenSrc.y - intrinsics(1,2)) / intrinsics(1,1), dSrc, 1.0);

//            if(currFrameNumber < 350)
//            {
//                dotSrc = Eigen::Vector4f(dSrc * (screenSrc.x - intrinsics(0,2)) / intrinsics(0,0),
//                                         dSrc * (screenSrc.y - intrinsics(1,2)) / intrinsics(1,1), dSrc, 1.0);
//
//            }
//            else{
//                dotSrc = last_state;
//
//            }
            //Eigen::Vector4f dotSrc_world = prev2world * dotSrc;


            float fu = bilinearInterpolationFloat(screenSrc.x, screenSrc.y, flowU, width, height);
            float fv = bilinearInterpolationFloat(screenSrc.x, screenSrc.y, flowV, width, height);
            float2 screenFlow = make_float2(screenSrc.x - fu, screenSrc.y - fv);
            if(screenFlow.x <= 1 || screenFlow.x >= width - 1 || screenFlow.y <= 1 || screenFlow.y >= height - 1)
            {
                continue;
            }
            uchar person = personMask[(uint)(screenFlow.x + 0.5) + (uint)(screenFlow.y + 0.5) * width];
            uchar cat = instanceMask[(uint)(screenFlow.x + 0.5) + (uint)(screenFlow.y + 0.5) * width];
            if(person != 255)
            {
                if((hasTrackingLabel && cat == g_objectCat) || !hasTrackingLabel)
                {
                    float dTar = bilinearInterpolationFloat(screenFlow.x, screenFlow.y, depTar, width, height);

                    if(!std::isinf(dTar) && dTar > 0) {

                        Eigen::Vector4f dotFlow = Eigen::Vector4f(dTar * (screenFlow.x - intrinsics(0,2)) / intrinsics(0,0),
                                                                  dTar * (screenFlow.y - intrinsics(1,2)) / intrinsics(1,1), dTar, 1.0);
                        //Eigen::Vector4f dotFlow_world = curr2world * dotFlow;
                        //std::cout<<"dTar:"<<dTar<<", dotFlow:"<<dotFlow.x()<<", "<<dotFlow.y()<<", "<<dotFlow.z()<<std::endl;
                        float dist = sqrt((dotFlow.x() - dotSrc.x()) * (dotFlow.x() - dotSrc.x()) +
                                          (dotFlow.y() - dotSrc.y()) * (dotFlow.y() - dotSrc.y()) +
                                          (dotFlow.z() - dotSrc.z()) * (dotFlow.z() - dotSrc.z()));
                        if(dist < 0.08)
                        {
                            cv::Point2f cvP2(screenFlow.x, screenFlow.y);
                            cv::circle(optimizedMat_tar, cvP2, 3, cv::Scalar(255, 0, 120), -1);

                            // get two poses
                            g2o::VertexSE3* vp0 =
                                    dynamic_cast<g2o::VertexSE3*>(optimizer_.vertices().find(0)->second);
                            g2o::VertexSE3* vp1 =
                                    dynamic_cast<g2o::VertexSE3*>(optimizer_.vertices().find(1)->second);

                            // calculate the relative 3D position of the point
                            //Eigen::Vector3d pt0, pt1;
                            //pt0 = vp0->estimate().inverse() * Eigen::Vector3d(dotSrc.x(),dotSrc.y(),dotSrc.z());
                            //pt1 = vp1->estimate().inverse() * Eigen::Vector3d(dotFlow.x(), dotFlow.y(), dotFlow.z());

                            g2o::Edge_V_V_GICP* e  // new edge with correct cohort for caching
                                    = new g2o::Edge_V_V_GICP();
                            e->setVertex(0, vp0);  // first viewpoint
                            e->setVertex(1, vp1);  // second viewpoint

                            g2o::EdgeGICP meas;
                            meas.pos0 = Eigen::Vector3d(dotSrc.x(),dotSrc.y(),dotSrc.z());
                            meas.pos1 = Eigen::Vector3d(dotFlow.x(), dotFlow.y(), dotFlow.z());
                            //meas.normal0 = nm0;
                            //meas.normal1 = nm1;

                            e->setMeasurement(meas);
                            //e->inverseMeasurement().pos() = -kp;

                            meas = e->measurement();
                            // use this for point-plane
                            e->information() = meas.prec0(0.01);
                            optimizer_.addEdge(e);
//                        EdgeFlowDepth* edgeFlowDepth = new EdgeFlowDepth(Eigen::Vector3d(last_state.x(),last_state.y(),last_state.z()));
//                        edgeFlowDepth->setVertex(0, pose);
//                        edgeFlowDepth->setMeasurement(Eigen::Vector3d(dotFlow_world.x(), dotFlow_world.y(), dotFlow_world.z()));
//                        edgeFlowDepth->setInformation(Eigen::Matrix3d::Identity());
//                        optimizer_.addEdge(edgeFlowDepth);
                        }


                    }
                }
            }
        }


        // move second cam off of its true position
        g2o::VertexSE3* vc =
                dynamic_cast<g2o::VertexSE3*>(optimizer_.vertices().find(1)->second);
        Eigen::Isometry3d cam = vc->estimate();
        cam.translation() = Eigen::Vector3d(0, 0, 0);
        vc->setEstimate(cam);

        optimizer_.initializeOptimization();
        optimizer_.computeActiveErrors();
        cout << "Initial chi2 = " << FIXED(optimizer_.chi2()) << endl;

        optimizer_.setVerbose(true);

        optimizer_.optimize(20);

        cout << endl << "Second vertex:" << endl;
        cout << dynamic_cast<g2o::VertexSE3*>(optimizer_.vertices().find(1)->second)
                ->estimate()
                .translation()
                .transpose()
             << endl;
        Eigen::Matrix4d transform_Curr = dynamic_cast<g2o::VertexSE3*>(optimizer_.vertices().find(1)->second)->estimate().matrix();

        //Eigen::Matrix4d transform_Curr = pose->estimate().matrix();
        cout << endl << "after optimization:" << endl;
        Eigen::Matrix4f transform_Curr4f = Eigen::Matrix4f();
        transform_Curr4f(0,0) = transform_Curr(0,0);
        transform_Curr4f(0,1) = transform_Curr(0,1);
        transform_Curr4f(0,2) = transform_Curr(0,2);
        transform_Curr4f(0,3) = transform_Curr(0,3);
        transform_Curr4f(1,0) = transform_Curr(1,0);
        transform_Curr4f(1,1) = transform_Curr(1,1);
        transform_Curr4f(1,2) = transform_Curr(1,2);
        transform_Curr4f(1,3) = transform_Curr(1,3);
        transform_Curr4f(2,0) = transform_Curr(2,0);
        transform_Curr4f(2,1) = transform_Curr(2,1);
        transform_Curr4f(2,2) = transform_Curr(2,2);
        transform_Curr4f(2,3) = transform_Curr(2,3);
        transform_Curr4f(3,0) = transform_Curr(3,0);
        transform_Curr4f(3,1) = transform_Curr(3,1);
        transform_Curr4f(3,2) = transform_Curr(3,2);
        transform_Curr4f(3,3) = transform_Curr(3,3);
        std::cout<<"transform_Curr4f:"<<transform_Curr4f<<std::endl;
        g_objPoses.push_back(transform_Curr4f.inverse() * g_objPoses[g_objPoses.size() - 1]);
        cv::imwrite("../test/"+std::to_string(currFrameNumber)+"_src.jpg", optimizedMat_src);
        cv::imwrite("../test/"+std::to_string(currFrameNumber)+"_tar.jpg", optimizedMat_tar);

        ofstream ofs;
        ofs.open("../objPose/"+std::to_string(currFrameNumber)+".txt",ios::out );
        ofs << g_objPoses[g_objPoses.size() - 1](0,0)<<" "<<g_objPoses[g_objPoses.size() - 1](0,1)<<" "<<g_objPoses[g_objPoses.size() - 1](0,2)<<" "<<g_objPoses[g_objPoses.size() - 1](0,3)<< " " <<
               g_objPoses[g_objPoses.size() - 1](1,0)<<" "<<g_objPoses[g_objPoses.size() - 1](1,1)<<" "<<g_objPoses[g_objPoses.size() - 1](1,2)<<" "<<g_objPoses[g_objPoses.size() - 1](1,3)<< " " <<
               g_objPoses[g_objPoses.size() - 1](2,0)<<" "<<g_objPoses[g_objPoses.size() - 1](2,1)<<" "<<g_objPoses[g_objPoses.size() - 1](2,2)<<" "<<g_objPoses[g_objPoses.size() - 1](2,3)<< " " <<
               g_objPoses[g_objPoses.size() - 1](3,0)<<" "<<g_objPoses[g_objPoses.size() - 1](3,1)<<" "<<g_objPoses[g_objPoses.size() - 1](3,2)<<" "<<g_objPoses[g_objPoses.size() - 1](3,3);

        ofs.close();
    }

    void genObjects(std::vector<Eigen::Vector4f> cloud, uchar cat)
    {
        g_objectCloud = cloud;
        g_objPoses.push_back(Eigen::Matrix4f::Identity());
        g_objectCat = cat;
    }
    int getObjects_cloud_num()
    {
        return g_objectCloud.size();
    }

   uchar* getDynamicMapCpu()
   {

   }











    float bilinearInterpolationFloat(float x, float y, const float* d_input, unsigned int imageWidth, unsigned int imageHeight)
    {

        const Eigen::Vector2i p00 = Eigen::Vector2i(floor(x), floor(y));
        const Eigen::Vector2i p01 = p00 + Eigen::Vector2i(0.0f, 1.0f);
        const Eigen::Vector2i p10 = p00 + Eigen::Vector2i(1.0f, 0.0f);
        const Eigen::Vector2i p11 = p00 + Eigen::Vector2i(1.0f, 1.0f);

        const float alpha = x - p00.x();
        const float beta = y - p00.y();

        float s0 = 0.0f; float w0 = 0.0f;
        if (p00.x() < imageWidth && p00.y() < imageHeight) { float v00 = d_input[p00.y()*imageWidth + p00.x()]; if (!std::isinf(v00)) { s0 += (1.0f - alpha)*v00; w0 += (1.0f - alpha); } }
        if (p10.x() < imageWidth && p10.y() < imageHeight) { float v10 = d_input[p10.y()*imageWidth + p10.x()]; if (!std::isinf(v10)) { s0 += alpha *v10; w0 += alpha; } }

        float s1 = 0.0f; float w1 = 0.0f;
        if (p01.x() < imageWidth && p01.y() < imageHeight) { float v01 = d_input[p01.y()*imageWidth + p01.x()]; if (!std::isinf(v01)) { s1 += (1.0f - alpha)*v01; w1 += (1.0f - alpha); } }
        if (p11.x() < imageWidth && p11.y() < imageHeight) { float v11 = d_input[p11.y()*imageWidth + p11.x()]; if (!std::isinf(v11)) { s1 += alpha *v11; w1 += alpha; } }

        const float p0 = s0 / w0;
        const float p1 = s1 / w1;

        float ss = 0.0f; float ww = 0.0f;
        if (w0 > 0.0f) { ss += (1.0f - beta)*p0; ww += (1.0f - beta); }
        if (w1 > 0.0f) { ss += beta *p1; ww += beta; }

        if (ww > 0.0f) return ss / ww;
        else		   return 0.0;
    }
    cv::Mat ucharToMat(const uchar4* p2, const int width, const int height)
    {
        //cout<< "length: " << p2-> << endl;
        int img_width = width;
        int img_height = height;
        cv::Mat img(cv::Size(img_width, img_height), CV_8UC3);
        for (int i = 0; i < img_width * img_height; i++)
        {
            int b = p2[i].x;
            int g = p2[i].y;
            int r = p2[i].z;

            img.at<cv::Vec3b>(i / img_width, (i % img_width))[0] = r;
            img.at<cv::Vec3b>(i / img_width, (i % img_width))[1] = g;
            img.at<cv::Vec3b>(i / img_width, (i % img_width))[2] = b;


        }
        return img;
    }
private:
    g2o::SparseOptimizer optimizer_;
    std::vector<Eigen::Vector4f> g_objectCloud; // init world coo
    uchar g_objectCat;
    std::vector<Eigen::Matrix4f> g_objPoses;
};

#endif //MAINLOOP_DYNAMICOBJECTMANAGER_H
