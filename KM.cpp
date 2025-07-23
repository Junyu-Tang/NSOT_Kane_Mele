#include <iostream>
#include <fstream>
#include <math.h>
#include <float.h>
#include <cmath>
#include <complex> 
#include <vector>
#include <chrono>
#include <omp.h>
#include <thread>
#include <stdexcept>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/KroneckerProduct>
#include <iomanip>
// clang++ -std=c++20 -I ./eigen3.4 -fopenmp KM.cpp -o KM_model  -O3

using namespace std;
using namespace std::chrono;
using Eigen::SelfAdjointEigenSolver;
using Eigen::kroneckerProduct;
using Eigen::MatrixXcf;
using Eigen::MatrixXf;
using Eigen::Matrix2cf;
using Eigen::Matrix4cf;
using Eigen::Matrix4f;
using Eigen::VectorXf;
using Eigen::VectorXcf;
using Eigen::Vector4cf;
using Eigen::Vector2f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using Eigen::all;
using Eigen::seq;

const float pi = 3.14159265f;
const complex<float>  u_i(0.0f, 1.0f);

void trackProgress(const std::atomic<int>& completedIterations, int totalIterations){
    int previousProgress = 0;
    while (completedIterations < totalIterations) {
        int currentProgress = (completedIterations.load(std::memory_order_relaxed) * 100) / totalIterations;
        if (currentProgress > previousProgress) {
            std::cout << "\rProgress: " << currentProgress << "%" << std::flush;
            previousProgress = currentProgress;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
}

void Write_Data(ofstream &file, MatrixXf data){
    int row = data.rows();
    int col = data.cols();
    for(int ii = 0; ii < row; ++ii){
        for(int jj = 0; jj < col; ++jj){
            file << data(ii, jj);
            if(jj < col - 1) file << ","; 
        }
        file << "\n";  
    }
}

void Write_Data(const char* path, const MatrixXf& data) {
    std::ofstream file(path);
    if (file.is_open()) {
        file << data.format(Eigen::IOFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n"));
        file.close();
    }
    else {
        std::cerr << "Unable to open file: " << path << std::endl;
    }
}

VectorXf linspace(float min, float max, int N=2) {
    VectorXf linspaced(N);
    float delta = (max - min) / (N - 1);
    for(int i = 0; i <= N-1; ++i) linspaced(i) = min + i * delta;
    return linspaced;
}

VectorXf LLG_solver(Vector3f M1, Vector3f M2, Vector3f S_1, Vector3f S_2, Vector3f Ans_axis, float Ans, float JAFM, float Bz, float Gilbert, float slope, float eta){ 
  //JAFM   :  AFM exchange coupling (>0)
  //Ans    :  Anisotropy (>0: easy axis, <0 hard axis)
  //Gilbert:  Gilbert damping constant (>0)
  //slope  :  Time increment (>0)
  //eta    :  SOT strength coefficient

  Vector3f M1_out;
  Vector3f M2_out;
  VectorXf out(6);
  MatrixXf Id  = MatrixXf::Identity(3, 3);

  // M1 sublattice
  Vector3f H1    = (Bz*Ans_axis + eta*S_1 - JAFM*M2 + 2*Ans*M1.dot(Ans_axis)*Ans_axis);  // Effective field: m1 x -H1
  Vector3f H_G1  = Gilbert*H1.cross(M1) / (1+Gilbert*Gilbert);                                                 // Gilbert damping field: G*(H1 x m) 
  Vector3f H_t1  = H_G1-H1;                                                                                    // Total Effective field: m1 x (H_G1 - H1)
  MatrixXf A1(3,3); A1 << 0, H_t1(2), -H_t1(1), -H_t1(2), 0, H_t1(0), H_t1(1), -H_t1(0), 0;
  MatrixXf temp11=(Id-A1*slope/2).inverse();
  MatrixXf temp12=(Id+A1*slope/2);
           M1_out=(temp11*temp12)*M1;

  // M2 sublattice
  Vector3f H2    = (Bz*Ans_axis + eta*S_2 - JAFM*M1 + 2*Ans*M2.dot(Ans_axis)*Ans_axis);   // Effective field: m1 x -H2
  Vector3f H_G2  = Gilbert*H2.cross(M2)/ (1+Gilbert*Gilbert);                                                                        // Gilbert damping field: G*(H1 x m) 
  Vector3f H_t2  = H_G2-H2 / (1+Gilbert*Gilbert);                                                               // Total Effective field: m2 x (H_G2 - H2)
  MatrixXf A2(3,3); A2 << 0, H_t2(2), -H_t2(1), -H_t2(2), 0, H_t2(0), H_t2(1), -H_t2(0), 0;
  MatrixXf temp21=(Id-A2*slope/2).inverse();
  MatrixXf temp22=(Id+A2*slope/2);
           M2_out=(temp21*temp22)*M2;

  // Output
  out << M1_out, M2_out;
  return out;
}

class Kane_Mele{
    public:
        //Paramters
        float a0, b0;
        float t, lso, lr, lv, Jsd;
        float mAx, mAy, mAz, mBx, mBy, mBz;
        int N1, N2, N3, N4;
        float theta1   = 0;
        float   phi1   = 0;
        float theta2   = pi;
        float   phi2   = 0;
        //Number of occupied bands
        int   n_occ    = 2;
        // Gyromagnetic ratio (unit: THz*rad/T for angular frequency)
        float   gyro   = 0.176;
        // Anisotropy, exchange coupling, Gilbert damping
        Vector3f Ans_axis; 
        float Ans, JAFM, Gilbert;
        // Pauli Matrices
        Matrix2cf S0, Sx, Sy, Sz;
        // Sublattice Matrices
        Matrix2cf sub_A, sub_B;
        // Spin on Sublattices
        Matrix4cf Ax, Ay, Az, Bx, By, Bz;
        // Gamma matrices in Hamiltonian
        Matrix4cf G1, G2, G3, G4, G5, G6, G7, G8;
        // Time-reversal operator
        Matrix4cf TR;

        //Constructor
        Kane_Mele(float init_a0 = 0.5, float init_t = 1.0){
            // Lattice geometry
            a0 = init_a0;
            t  = init_t;
            lso = 0.05*t;
            lr  = 0.0*t;
            lv  = 0.05*t;
            Jsd = 0.1*t;
            b0 = 4*pi/(sqrtf(3.0f)*a0);
            // Each time the Hamiltonian is called, magnetization should be set before.
            mAx = sin(theta1)*cos(phi1); mBx = sin(theta2)*cos(phi2);
            mAy = sin(theta1)*sin(phi1); mBy = sin(theta2)*sin(phi2);
            mAz = cos(theta1);           mBz = cos(theta2); 
            // Anisotropy axis
            Ans_axis << 0, 0, 1;
            // Amplitude of anisotropy (>0: easy axis, <0: hard axis. Unit: THz for angular frequency)
            Ans     = gyro*250/10000;
            // Exchange couplings (>0. Unit: THz for angular frequency)
            JAFM    = gyro*50;
            // Dimensionless Gilber tdamping
            Gilbert = 0.01;      
            // Pauli Matrices
            S0 << 1,0,0,1;
            Sx << 0,1,1,0;
            Sy << 0,-u_i,u_i,0;
            Sz << 1,0,0,-1;
            // Sublattice Matrices
            sub_A << 1,0,0,0;
            sub_B << 0,0,0,1;
            //sub_A << 0,1,1,0;
            //sub_B << 0,-u_i,u_i,0;

            // Spin on Sublattices
            Ax = kroneckerProduct(sub_A,Sx);
            Ay = kroneckerProduct(sub_A,Sy);
            Az = kroneckerProduct(sub_A,Sz);
            Bx = kroneckerProduct(sub_B,Sx);
            By = kroneckerProduct(sub_B,Sy);
            Bz = kroneckerProduct(sub_B,Sz);
            // Gamma matrices in Hamiltonian
            G1 = kroneckerProduct(Sx,S0);
            G2 = kroneckerProduct(Sz,S0);
            G3 = kroneckerProduct(Sy,Sx);
            G4 = kroneckerProduct(Sy,Sy);
            G5 = -u_i*(G1*G2-G2*G1)/2;
            G6 = -u_i*(G1*kroneckerProduct(Sy,Sz)-kroneckerProduct(Sy,Sz)*G1)/2;
            G7 = -u_i*(G2*G3-G3*G2)/2;
            G8 = -u_i*(G2*G4-G4*G2)/2;
            // Time-reversal operator
            TR = u_i*kroneckerProduct(S0,Sy);                
        }

        void set_M(float t1, float p1, float t2, float p2){
            theta1 = t1; theta2 = t2; phi1 = p1; phi2 = p2;
            mAx = sin(t1)*cos(p1); mBx = sin(t2)*cos(p2);
            mAy = sin(t1)*sin(p1); mBy = sin(t2)*sin(p2);
            mAz = cos(t1);         mBz = cos(t2); 
        }

        void set_M(Vector3f M1, Vector3f M2){
            mAx = M1(0); mBx = M2(0);
            mAy = M1(1); mBy = M2(1);
            mAz = M1(2); mBz = M2(2); 
        }

        Vector2f BZ_map(int ii,int jj, VectorXf b1, VectorXf b2){
            Vector2f kxy;
            kxy << (b1(ii)-b2(jj))*cos(pi/6), (b1(ii)+b2(jj))*sin(pi/6);  
            return kxy;
        }

        float Dirac_delta(float Gamma, float En, float Ef){
          float numerator = 2*Gamma*Gamma*Gamma;
          float denominator = pi*((En-Ef)*(En-Ef)+Gamma*Gamma)*((En-Ef)*(En-Ef)+Gamma*Gamma);
          return numerator/denominator;
        }

        float broadening(float Gamma, float En, float Em){
          float numerator = (En-Em)*(En-Em)-Gamma*Gamma;
          float denominator = ((En-Em)*(En-Em)+Gamma*Gamma)*((En-Em)*(En-Em)+Gamma*Gamma);
          return numerator/denominator;
        }

        MatrixXcf S_matrix(MatrixXcf vecn, MatrixXcf vecm, int n_occ){
            MatrixXcf S(n_occ, n_occ);
            for(int ii = 0; ii < n_occ; ii++){
              for(int jj = 0; jj < n_occ; jj++){
                  S(ii,jj) = vecn.col(ii).dot(vecm.col(jj));
              }
            }
            return S;
        }

        complex<float> Pfaffian(float qx, float qy, int n_occ){
            Matrix4cf Eigen_vec;
            SelfAdjointEigenSolver<MatrixXcf> solver;
            solver.compute(Hamiltonian(qx, qy));
            Eigen_vec = solver.eigenvectors();
            Matrix4cf theta_vec = TR*(Eigen_vec.conjugate());
            MatrixXcf sewing = S_matrix(Eigen_vec, theta_vec, n_occ);
            return sewing(0,1);
        }

        Matrix4cf Hamiltonian(float qx, float qy){
          float x   = qx*a0/2;
          float y   = sqrtf(3.0f)*qy*a0/2;
          float d1  = t*(1+2*cos(x)*cos(y));
          float d2  = lv;
          float d3  = lr*(1-cos(x)*cos(y));
          float d4  = -sqrtf(3.0f)*lr*sin(x)*sin(y);
          float d5  = -2*t*cos(x)*sin(y);
          float d6  = lso*(2*sin(2*x)-4*sin(x)*cos(y));
          float d7  = -lr*cos(x)*sin(y);
          float d8  = sqrtf(3.0f)*lr*sin(x)*cos(y);
          // call set_M(...) before Hamiltonian is called          
          Matrix4cf Hm = MatrixXcf::Zero(4,4);
          Hm  += Jsd*(kroneckerProduct(sub_A,mAx*Sx)+kroneckerProduct(sub_A,mAy*Sy)+kroneckerProduct(sub_A,mAz*Sz));
          Hm  += Jsd*(kroneckerProduct(sub_B,mBx*Sx)+kroneckerProduct(sub_B,mBy*Sy)+kroneckerProduct(sub_B,mBz*Sz));
          return d1*G1+d2*G2+d3*G3+d4*G4+d5*G5+d6*G6+d7*G7+d8*G8+Hm;
        }

        Matrix4cf Hkx(float qx, float qy){
          float x   = qx*a0/2;
          float y   = sqrtf(3.0f)*qy*a0/2;
          float d1  = t*(-2*sin(x)*cos(y));
          float d3  = lr*(sin(x)*cos(y));
          float d4  = -sqrtf(3.0f)*lr*cos(x)*sin(y);
          float d5  = 2*t*sin(x)*sin(y);
          float d6  = lso*(4*cos(2*x)-4*cos(x)*cos(y));
          float d7  = lr*sin(x)*sin(y);
          float d8  = sqrtf(3.0f)*lr*cos(x)*cos(y);
          return a0*(d1*G1+d3*G3+d4*G4+d5*G5+d6*G6+d7*G7+d8*G8)/2;
        }

        Matrix4cf Hky(float qx, float qy){
          float x   = qx*a0/2;
          float y   = sqrtf(3.0f)*qy*a0/2;
          float d1  = t*(-2*cos(x)*sin(y));
          float d3  = lr*(cos(x)*sin(y));
          float d4  = -sqrtf(3.0f)*lr*sin(x)*cos(y);
          float d5  = -2*t*cos(x)*cos(y);
          float d6  = lso*4*sin(x)*sin(y);
          float d7  = -lr*cos(x)*cos(y);
          float d8  = -sqrtf(3.0f)*lr*sin(x)*sin(y);
          return sqrtf(3.0f)*a0*(d1*G1+d3*G3+d4*G4+d5*G5+d6*G6+d7*G7+d8*G8)/2;
        }

        float Chern_number_Wilson(int n1, int n2){
            VectorXf q1 = linspace(0, b0, n1);
            VectorXf q2 = linspace(0, b0, n2);
            Vector2f v1, v2, v3, v4;
            MatrixXcf U14, U43, U32, U21;
            Matrix4cf Eigen_vec1, Eigen_vec2, Eigen_vec3, Eigen_vec4;
            MatrixXf F14321(n1-1, n2-1);
            SelfAdjointEigenSolver<MatrixXcf> solver;

            for(int ii = 0; ii < n1-1 ; ii++){
                for(int jj = 0; jj < n2-1 ; jj++){
                    v1 = BZ_map(ii,jj,q1,q2);
                    v2 = BZ_map(ii+1,jj,q1,q2);
                    v3 = BZ_map(ii+1,jj+1,q1,q2);
                    v4 = BZ_map(ii,jj+1,q1,q2);  

                    solver.compute(Hamiltonian(v1(0), v1(1)));
                    Eigen_vec1 = solver.eigenvectors();

                    solver.compute(Hamiltonian(v2(0), v2(1)));
                    Eigen_vec2 = solver.eigenvectors();

                    solver.compute(Hamiltonian(v3(0), v3(1)));
                    Eigen_vec3 = solver.eigenvectors();

                    solver.compute(Hamiltonian(v4(0), v4(1)));
                    Eigen_vec4 = solver.eigenvectors();  

                    U14 = S_matrix(Eigen_vec1, Eigen_vec4, n_occ);
                    U43 = S_matrix(Eigen_vec4, Eigen_vec3, n_occ);
                    U32 = S_matrix(Eigen_vec3, Eigen_vec2, n_occ);
                    U21 = S_matrix(Eigen_vec2, Eigen_vec1, n_occ);
                    
                    F14321(ii,jj) = imag(log((U14*U43*U32*U21).determinant()));
                }
            }
            return (F14321.sum())/(2.0*pi);
        }

        float Chern_number_Feynman(int n1, int n2, float Gamma=0){
            Matrix4cf Vec, Vx, Vy;
            Vector4f  Val;
            Vector2f kpoints;
            VectorXf q1 = linspace(0, b0, n1);
            VectorXf q2 = linspace(0, b0, n2);
            complex<float> Berry;
            SelfAdjointEigenSolver<MatrixXcf> solver;

            MatrixXf Bkxky(n1-1, n2-1);
            for(int ii = 0; ii < n1-1 ; ii++){
                for(int jj = 0; jj < n2-1 ; jj++){
                    kpoints = BZ_map(ii,jj,q1,q2);
                    solver.compute(Hamiltonian(kpoints(0), kpoints(1)));
                    Val = solver.eigenvalues();
                    Vec = solver.eigenvectors();

                    Vx = Vec.adjoint()*Hkx(kpoints(0), kpoints(1))*Vec;
                    Vy = Vec.adjoint()*Hky(kpoints(0), kpoints(1))*Vec;

                    Berry  = Vx(0,2)*Vy(2,0)*broadening(Gamma, Val(0), Val(2));
                    Berry += Vx(0,3)*Vy(3,0)*broadening(Gamma, Val(0), Val(3));
                    Berry += Vx(1,2)*Vy(2,1)*broadening(Gamma, Val(1), Val(2));
                    Berry += Vx(1,3)*Vy(3,1)*broadening(Gamma, Val(1), Val(3));
                    
                    Bkxky(ii,jj) = -2*Berry.imag()/(2*pi);
                }
            }
           return Bkxky.sum()*b0*b0*sqrtf(3.0f)/(2*(n1-1)*(n2-1));
        }

        float Spin_Chern_number(int n1, int n2, float Gamma=0){
            Matrix4cf Vec, Vx, Vy;
            Vector4f  Val;
            Vector2f kpoints;
            VectorXf q1 = linspace(0, b0, n1);
            VectorXf q2 = linspace(0, b0, n2);
            complex<float> Berry;
            SelfAdjointEigenSolver<MatrixXcf> solver;
            Matrix4cf spinz = kroneckerProduct(S0, Sz);
            Matrix4cf Is;

            MatrixXf Bkxky(n1-1, n2-1);
            for(int ii = 0; ii < n1-1 ; ii++){
                for(int jj = 0; jj < n2-1 ; jj++){
                    kpoints = BZ_map(ii,jj,q1,q2);
                    solver.compute(Hamiltonian(kpoints(0), kpoints(1)));
                    Val = solver.eigenvalues();
                    Vec = solver.eigenvectors();
                    Is  = (Hkx(kpoints(0), kpoints(1))*spinz + spinz*Hkx(kpoints(0), kpoints(1))) / 4;

                    Vx = Vec.adjoint()*Is*Vec;
                    Vy = Vec.adjoint()*Hky(kpoints(0), kpoints(1))*Vec;

                    Berry  = Vx(0,2)*Vy(2,0)*broadening(Gamma, Val(0), Val(2));
                    Berry += Vx(0,3)*Vy(3,0)*broadening(Gamma, Val(0), Val(3));
                    Berry += Vx(1,2)*Vy(2,1)*broadening(Gamma, Val(1), Val(2));
                    Berry += Vx(1,3)*Vy(3,1)*broadening(Gamma, Val(1), Val(3));
                    
                    Bkxky(ii,jj) = -2*Berry.imag()/(2*pi);
                }
            }
           return Bkxky.sum()*b0*b0*sqrtf(3.0f)/(2*(n1-1)*(n2-1));
        }

        int Z2_number(int n1, int n2){
            if(n1%2 == 0 || n2%2 == 0) throw runtime_error("Meshgrid points must be odd!");
            if(Jsd != 0) throw runtime_error("Magnetic term must be turned off to calculate Z2 number!");

            VectorXf q1 = linspace(0, b0/2, n1);//Half BZ
            VectorXf q2 = linspace(0, b0,   n2);
            Vector2f v1, v2, v3, v4;
            MatrixXcf U14, U43, U32, U21;
            Matrix4cf Eigen_vec1, Eigen_vec2, Eigen_vec3, Eigen_vec4;
            MatrixXf F14321(n1-1, n2-1);
            SelfAdjointEigenSolver<MatrixXcf> solver;

            for(int ii = 0; ii < n1-1 ; ii++){
                for(int jj = 0; jj < n2-1 ; jj++){
                    v1 = BZ_map(ii,jj,q1,q2);
                    v2 = BZ_map(ii+1,jj,q1,q2);
                    v3 = BZ_map(ii+1,jj+1,q1,q2);
                    v4 = BZ_map(ii,jj+1,q1,q2);  

                    solver.compute(Hamiltonian(v1(0), v1(1)));
                    Eigen_vec1 = solver.eigenvectors();

                    solver.compute(Hamiltonian(v2(0), v2(1)));
                    Eigen_vec2 = solver.eigenvectors();

                    solver.compute(Hamiltonian(v3(0), v3(1)));
                    Eigen_vec3 = solver.eigenvectors();

                    solver.compute(Hamiltonian(v4(0), v4(1)));
                    Eigen_vec4 = solver.eigenvectors();  

                    U14 = S_matrix(Eigen_vec1, Eigen_vec4, n_occ);
                    U43 = S_matrix(Eigen_vec4, Eigen_vec3, n_occ);
                    U32 = S_matrix(Eigen_vec3, Eigen_vec2, n_occ);
                    U21 = S_matrix(Eigen_vec2, Eigen_vec1, n_occ);
                    
                    F14321(ii,jj) = imag(log((U14*U43*U32*U21).determinant()));
                }
            }

            VectorXcf PT1 = VectorXcf::Ones((n2-1)/2);
            for(int jj = 0; jj<(n2-1)/2; jj++){
                v1 = BZ_map(0,jj,q1,q2);
                v4 = BZ_map(0,jj+1,q1,q2);

                solver.compute(Hamiltonian(v1(0), v1(1)));
                Eigen_vec1 = solver.eigenvectors();

                solver.compute(Hamiltonian(v4(0), v4(1)));
                Eigen_vec4 = solver.eigenvectors();

                PT1(jj) = S_matrix(Eigen_vec4, Eigen_vec1, n_occ).determinant();
            }
            
            v1 = BZ_map(0,(n2+1)/2-1,q1,q2);
            v4 = BZ_map(0,0,q1,q2);
            float PK1 = imag(log((PT1.prod())*Pfaffian(v4(0), v4(1), n_occ)/Pfaffian(v1(0), v1(1), n_occ)));

            VectorXcf PT2 = VectorXcf::Ones((n2-1)/2);
            for(int jj = 0; jj<(n2-1)/2; jj++){
                v2 = BZ_map(n1-1,jj,q1,q2);
                v3 = BZ_map(n1-1,jj+1,q1,q2);

                solver.compute(Hamiltonian(v2(0), v2(1)));
                Eigen_vec2 = solver.eigenvectors();

                solver.compute(Hamiltonian(v3(0), v3(1)));
                Eigen_vec3 = solver.eigenvectors();

                PT2(jj) = S_matrix(Eigen_vec3, Eigen_vec2, n_occ).determinant();
            }
            
            v2 = BZ_map(n1-1,(n2+1)/2-1,q1,q2);
            v3 = BZ_map(n1-1,0,q1,q2);
            float PK2 = imag(log((PT2.prod())*Pfaffian(v3(0), v3(1), n_occ)/Pfaffian(v2(0), v2(1), n_occ)));

            return abs(int(round((F14321.sum()+2*PK1-2*PK2)/(2*pi))) % 2);
        }

        VectorXf Noneq_spin_surface(int n1, int n2, float Gamma, float Ef, float E_phi){
            VectorXf  spin(6);
            Matrix4cf Vec;
            Vector4f  Val, Ed;
            Vector2f kpoints;
            Vector4cf SAx, SAy, SAz, SBx, SBy, SBz, Vx, Vy;
            VectorXf q1 = linspace(0, b0, n1);
            VectorXf q2 = linspace(0, b0, n2);
            SelfAdjointEigenSolver<MatrixXcf> solver;

            MatrixXf FAx(n1-1, n2-1), FAy(n1-1, n2-1), FAz(n1-1, n2-1), FBx(n1-1, n2-1), FBy(n1-1, n2-1), FBz(n1-1, n2-1);
            for(int ii = 0; ii < n1-1 ; ii++){
                for(int jj = 0; jj < n2-1 ; jj++){
                    kpoints = BZ_map(ii,jj,q1,q2);
                    solver.compute(Hamiltonian(kpoints(0), kpoints(1)));
                    Val = solver.eigenvalues();
                    Vec = solver.eigenvectors();

                    Matrix4f Ed = Matrix4f(Vector4f(Dirac_delta(Gamma, Val(0), Ef), Dirac_delta(Gamma, Val(1), Ef), Dirac_delta(Gamma, Val(2), Ef), Dirac_delta(Gamma, Val(3), Ef)).asDiagonal());
                    SAx = (Vec.adjoint()*Ax*Vec*Ed).diagonal();
                    SAy = (Vec.adjoint()*Ay*Vec*Ed).diagonal();
                    SAz = (Vec.adjoint()*Az*Vec*Ed).diagonal();
                    SBx = (Vec.adjoint()*Bx*Vec*Ed).diagonal();
                    SBy = (Vec.adjoint()*By*Vec*Ed).diagonal();
                    SBz = (Vec.adjoint()*Bz*Vec*Ed).diagonal();

                    Vx = (Vec.adjoint()*Hkx(kpoints(0), kpoints(1))*Vec).diagonal();
                    Vy = (Vec.adjoint()*Hky(kpoints(0), kpoints(1))*Vec).diagonal();

                    FAx(ii,jj) =  cos(E_phi)*((SAx.dot(Vx)).real()) + sin(E_phi)*((SAx.dot(Vy)).real());
                    FAy(ii,jj) =  cos(E_phi)*((SAy.dot(Vx)).real()) + sin(E_phi)*((SAy.dot(Vy)).real());
                    FAz(ii,jj) =  cos(E_phi)*((SAz.dot(Vx)).real()) + sin(E_phi)*((SAz.dot(Vy)).real());
                    FBx(ii,jj) =  cos(E_phi)*((SBx.dot(Vx)).real()) + sin(E_phi)*((SBx.dot(Vy)).real());
                    FBy(ii,jj) =  cos(E_phi)*((SBy.dot(Vx)).real()) + sin(E_phi)*((SBy.dot(Vy)).real());
                    FBz(ii,jj) =  cos(E_phi)*((SBz.dot(Vx)).real()) + sin(E_phi)*((SBz.dot(Vy)).real());

                }
            }
            spin << FAx.sum(), FAy.sum(), FAz.sum(), FBx.sum(), FBy.sum(), FBz.sum();
            return spin*b0*b0*sin(pi/3)/((n1-1)*(n2-1));
            /* Unit: hbar/2 per unit cell                              (unit cell area) * (nm/µm) / (residual coefficient)
            For electric driving field |E| = 1V/µm, the coefficient is a0^2 * sqrt(3)/2 * 10^-3   / (2*(2*pi)^2*Gamma)
            */
        }

        VectorXf Noneq_spin_sea(int n1, int n2, float Gamma, float E_phi){
            VectorXf  spin(6);
            Matrix4cf Vec, SAx, SAy, SAz, SBx, SBy, SBz, Vx, Vy;
            Vector4f  Val;
            Vector2f kpoints;
            VectorXf q1 = linspace(0, b0, n1);
            VectorXf q2 = linspace(0, b0, n2);
            complex<float> BAxVx, BAxVy, BBxVx, BBxVy, BAyVx, BAyVy, BByVx, BByVy, BAzVx, BAzVy, BBzVx, BBzVy;
            SelfAdjointEigenSolver<MatrixXcf> solver;

            MatrixXf FAx(n1-1, n2-1), FAy(n1-1, n2-1), FAz(n1-1, n2-1), FBx(n1-1, n2-1), FBy(n1-1, n2-1), FBz(n1-1, n2-1);

            #pragma omp parallel for private(kpoints, solver, Val, Vec, SAx, SAy, SAz, SBx, SBy, SBz, Vx, Vy, BAxVx, BAxVy, BBxVx, BBxVy, BAyVx, BAyVy, BByVx, BByVy, BAzVx, BAzVy, BBzVx, BBzVy)
            for(int ii = 0; ii < n1-1 ; ii++){
                for(int jj = 0; jj < n2-1 ; jj++){
                    kpoints = BZ_map(ii,jj,q1,q2);
                    solver.compute(Hamiltonian(kpoints(0), kpoints(1)));
                    Val = solver.eigenvalues();
                    Vec = solver.eigenvectors();

                    SAx = Vec.adjoint()*Ax*Vec;
                    SAy = Vec.adjoint()*Ay*Vec;
                    SAz = Vec.adjoint()*Az*Vec;
                    SBx = Vec.adjoint()*Bx*Vec;
                    SBy = Vec.adjoint()*By*Vec;
                    SBz = Vec.adjoint()*Bz*Vec;
                    Vx =  Vec.adjoint()*Hkx(kpoints(0), kpoints(1))*Vec;
                    Vy =  Vec.adjoint()*Hky(kpoints(0), kpoints(1))*Vec;

                    BAxVx  = SAx(0,2)*Vx(2,0)*broadening(Gamma, Val(0), Val(2));
                    BAxVx += SAx(0,3)*Vx(3,0)*broadening(Gamma, Val(0), Val(3));
                    BAxVx += SAx(1,2)*Vx(2,1)*broadening(Gamma, Val(1), Val(2));
                    BAxVx += SAx(1,3)*Vx(3,1)*broadening(Gamma, Val(1), Val(3));
                    BAxVy  = SAx(0,2)*Vy(2,0)*broadening(Gamma, Val(0), Val(2));
                    BAxVy += SAx(0,3)*Vy(3,0)*broadening(Gamma, Val(0), Val(3));
                    BAxVy += SAx(1,2)*Vy(2,1)*broadening(Gamma, Val(1), Val(2));
                    BAxVy += SAx(1,3)*Vy(3,1)*broadening(Gamma, Val(1), Val(3));

                    BBxVx  = SBx(0,2)*Vx(2,0)*broadening(Gamma, Val(0), Val(2));
                    BBxVx += SBx(0,3)*Vx(3,0)*broadening(Gamma, Val(0), Val(3));
                    BBxVx += SBx(1,2)*Vx(2,1)*broadening(Gamma, Val(1), Val(2));
                    BBxVx += SBx(1,3)*Vx(3,1)*broadening(Gamma, Val(1), Val(3));
                    BBxVy  = SBx(0,2)*Vy(2,0)*broadening(Gamma, Val(0), Val(2));
                    BBxVy += SBx(0,3)*Vy(3,0)*broadening(Gamma, Val(0), Val(3));
                    BBxVy += SBx(1,2)*Vy(2,1)*broadening(Gamma, Val(1), Val(2));
                    BBxVy += SBx(1,3)*Vy(3,1)*broadening(Gamma, Val(1), Val(3));

                    BAyVx  = SAy(0,2)*Vx(2,0)*broadening(Gamma, Val(0), Val(2));
                    BAyVx += SAy(0,3)*Vx(3,0)*broadening(Gamma, Val(0), Val(3));
                    BAyVx += SAy(1,2)*Vx(2,1)*broadening(Gamma, Val(1), Val(2));
                    BAyVx += SAy(1,3)*Vx(3,1)*broadening(Gamma, Val(1), Val(3));
                    BAyVy  = SAy(0,2)*Vy(2,0)*broadening(Gamma, Val(0), Val(2));
                    BAyVy += SAy(0,3)*Vy(3,0)*broadening(Gamma, Val(0), Val(3));
                    BAyVy += SAy(1,2)*Vy(2,1)*broadening(Gamma, Val(1), Val(2));
                    BAyVy += SAy(1,3)*Vy(3,1)*broadening(Gamma, Val(1), Val(3));

                    BByVx  = SBy(0,2)*Vx(2,0)*broadening(Gamma, Val(0), Val(2));
                    BByVx += SBy(0,3)*Vx(3,0)*broadening(Gamma, Val(0), Val(3));
                    BByVx += SBy(1,2)*Vx(2,1)*broadening(Gamma, Val(1), Val(2));
                    BByVx += SBy(1,3)*Vx(3,1)*broadening(Gamma, Val(1), Val(3));
                    BByVy  = SBy(0,2)*Vy(2,0)*broadening(Gamma, Val(0), Val(2));
                    BByVy += SBy(0,3)*Vy(3,0)*broadening(Gamma, Val(0), Val(3));
                    BByVy += SBy(1,2)*Vy(2,1)*broadening(Gamma, Val(1), Val(2));
                    BByVy += SBy(1,3)*Vy(3,1)*broadening(Gamma, Val(1), Val(3));

                    BAzVx  = SAz(0,2)*Vx(2,0)*broadening(Gamma, Val(0), Val(2));
                    BAzVx += SAz(0,3)*Vx(3,0)*broadening(Gamma, Val(0), Val(3));
                    BAzVx += SAz(1,2)*Vx(2,1)*broadening(Gamma, Val(1), Val(2));
                    BAzVx += SAz(1,3)*Vx(3,1)*broadening(Gamma, Val(1), Val(3));
                    BAzVy  = SAz(0,2)*Vy(2,0)*broadening(Gamma, Val(0), Val(2));
                    BAzVy += SAz(0,3)*Vy(3,0)*broadening(Gamma, Val(0), Val(3));
                    BAzVy += SAz(1,2)*Vy(2,1)*broadening(Gamma, Val(1), Val(2));
                    BAzVy += SAz(1,3)*Vy(3,1)*broadening(Gamma, Val(1), Val(3));

                    BBzVx  = SBz(0,2)*Vx(2,0)*broadening(Gamma, Val(0), Val(2));
                    BBzVx += SBz(0,3)*Vx(3,0)*broadening(Gamma, Val(0), Val(3));
                    BBzVx += SBz(1,2)*Vx(2,1)*broadening(Gamma, Val(1), Val(2));
                    BBzVx += SBz(1,3)*Vx(3,1)*broadening(Gamma, Val(1), Val(3));
                    BBzVy  = SBz(0,2)*Vy(2,0)*broadening(Gamma, Val(0), Val(2));
                    BBzVy += SBz(0,3)*Vy(3,0)*broadening(Gamma, Val(0), Val(3));
                    BBzVy += SBz(1,2)*Vy(2,1)*broadening(Gamma, Val(1), Val(2));
                    BBzVy += SBz(1,3)*Vy(3,1)*broadening(Gamma, Val(1), Val(3));
                    
                    FAx(ii,jj) = cos(E_phi)*BAxVx.imag() + sin(E_phi)*BAxVy.imag();
                    FBx(ii,jj) = cos(E_phi)*BBxVx.imag() + sin(E_phi)*BBxVy.imag();
                    FAy(ii,jj) = cos(E_phi)*BAyVx.imag() + sin(E_phi)*BAyVy.imag();
                    FBy(ii,jj) = cos(E_phi)*BByVx.imag() + sin(E_phi)*BByVy.imag();
                    FAz(ii,jj) = cos(E_phi)*BAzVx.imag() + sin(E_phi)*BAzVy.imag();
                    FBz(ii,jj) = cos(E_phi)*BBzVx.imag() + sin(E_phi)*BBzVy.imag();

                }
            }
            spin << FAx.sum(), FAy.sum(), FAz.sum(), FBx.sum(), FBy.sum(), FBz.sum();
            return spin*b0*b0*sqrtf(3.0f)/(2*(n1-1)*(n2-1)); //(step^2 * sqrt(3)/2)

            /* Unit: hbar/2 per unit cell                              (unit cell area) * (nm/µm) / (residual coefficient)
            For electric driving field |E| = 1V/µm, the coefficient is a0^2 * sqrt(3)/2 * 10^-3   / ((2*pi)^2)
            */
        }

        float Get_global_bandgap(int n1, int n2){
            VectorXf q1 = linspace(0, b0, n1);
            VectorXf q2 = linspace(0, b0, n2);
            Vector4f Val;
            Vector2f kpoints;
            SelfAdjointEigenSolver<MatrixXcf> solver;

            MatrixXf Bandgap(n1, n2);
            for(int ii = 0; ii < n1 ; ii++){
                for(int jj = 0; jj < n2 ; jj++){
                    kpoints = BZ_map(ii,jj,q1,q2);
                    solver.compute(Hamiltonian(kpoints(0), kpoints(1)), Eigen::EigenvaluesOnly);
                    Val = solver.eigenvalues();
                    Bandgap(ii,jj) = Val(2)-Val(1);
                }
            }
            return Bandgap.minCoeff();
        }


};


int main(){
    cout << "----------------------------------" << endl;
    cout << "Kane-Mele model with AFM order" << endl;

    auto start   = high_resolution_clock::now();
    float a0     = 0.5; // Lattice constant (nm)
    float t      = 1.0; // Hopping strength (eV)
    int N1       = 1001;
    int N2       = 1001;
    int N3       = 200;
    int N4       = 200;
    float theta1 = 0;
    float phi1   = 0;
    float theta2 = pi;
    float phi2   = 0;
    float Gamma  = 0.02*t; // Disorder strength (eV), default: 20 meV;

    cout << "----------------------------------"  << endl;
    cout << "Initializing the Kane-Mele Model..." << endl;
    Kane_Mele KM(a0, t);
    KM.set_M(theta1, phi1, theta2, phi2);


    cout << "KM Model successfully constructed!"     << endl;
    cout << "Initialized parameters:"                << endl;
    cout << "a0           = " << KM.a0 << "(nm)"     << endl;
    cout << "t            = " << KM.t  << "(eV)"     << endl;
    cout << "lso          = " << KM.lso << "(eV)"    << endl;
    cout << "lr           = " << KM.lr  << "(eV)"    << endl;
    cout << "lv           = " << KM.lv  << "(eV)"    << endl;
    cout << "Jsd          = " << KM.Jsd  << "(eV)"   << endl;
    cout << "Disorder     = " << Gamma << "(eV) "    << endl;
    cout << "theta1       = " << KM.theta1           << endl;
    cout << "phi1         = " << KM.phi1             << endl;
    cout << "theta2       = " << KM.theta2           << endl;
    cout << "phi2         = " << KM.phi2             << endl;
    cout << "Jex          = " << KM.JAFM << "(THz)"  << endl;
    cout << "Anisotropy   = " << KM.Ans  << "(THz)"  << endl;
    cout << "Gilbert      = " << KM.Gilbert          << endl;
    cout << "N1           = " << N1                  << endl;
    cout << "N2           = " << N2                  << endl;
    cout << "N3           = " << N3                  << endl;
    cout << "N4           = " << N4                  << endl;
    cout << "----------------------------------"     << endl;

    /*
    KM.lv=0.05*t;;
    KM.lso=0.05*t;;
    KM.lr=0.01*t;
    VectorXf temp(6);
    temp = KM.Noneq_spin_sea(N1, N2,Gamma,0);
    cout << "Spin Chern number Cs is " << KM.Spin_Chern_number(N1,N2) << endl;
    cout << "Chern number C is " << KM.Chern_number_Wilson(N1,N2) << endl;
    cout << "The neq_S are:" << endl;
    cout << temp*a0*a0*sqrt(3)/(2*(2*pi)*(2*pi)*1000) << endl;
    */
    


    
    cout << "Begin sweeping the parameters..."  << endl;
    VectorXf Param1 = linspace(-0.05*t, 0.05*t, N3);
    VectorXf Noneq_Spin_Ax(N3), Noneq_Spin_Ay(N3), Noneq_Spin_Az(N3), Noneq_Spin_Bx(N3), Noneq_Spin_By(N3), Noneq_Spin_Bz(N3);
    VectorXf Chern_number(N3);
    VectorXf Spin_Chern(N3);
    VectorXf temp(6);
    std::atomic<int> completedIterations(0);
    std::thread progressThread(trackProgress, std::ref(completedIterations), N3);
    #pragma omp parallel for private(KM, temp)
    for (int ii=0; ii<N3; ii++){
        KM.lv=0.04*t;
        KM.lr=0.02*t;
        KM.lso = Param1(ii);
        
        temp = KM.Noneq_spin_sea(N1, N2, Gamma,0);
        temp = temp*a0*a0*sqrt(3)/(2*(2*pi)*(2*pi)*1000);
        Noneq_Spin_Ax(ii) = temp(0);
        Noneq_Spin_Bx(ii) = temp(3);
        Chern_number(ii)  = KM.Chern_number_Feynman(N1, N2, 0);
        Spin_Chern(ii)    = KM.Spin_Chern_number(N1, N2, 0);

        completedIterations.fetch_add(1, std::memory_order_relaxed);
    }
    progressThread.join(); 
    cout << "\rProgress: " << 100 << "% (Fnished!)\n" << std::flush;
    cout << "Begin writing the data..."           << endl;
    cout << "----------------------------------"  << endl;
    const char *path_C="/Users/junyutang/Documents/cpp/projects/Kane_Mele_M/data/test/C(lso,lv=0.04,lr=0.02).csv";
    const char *path_Cs="/Users/junyutang/Documents/cpp/projects/Kane_Mele_M/data/test/Cs(lso,lv=0.04,lr=0.02).csv";
    const char *path_SAx="/Users/junyutang/Documents/cpp/projects/Kane_Mele_M/data/test/SAx(lso,lv=0.04,lr=0.02).csv";
    const char *path_SBx="/Users/junyutang/Documents/cpp/projects/Kane_Mele_M/data/test/SBx(lso,lv=0.04,lr=0.02).csv";
    ofstream file_C; file_C.open(path_C);
    ofstream file_Cs; file_Cs.open(path_Cs);
    ofstream file_SAx; file_SAx.open(path_SAx);
    ofstream file_SBx; file_SBx.open(path_SBx);
    Write_Data(file_C, Chern_number);
    Write_Data(file_Cs, Spin_Chern);
    Write_Data(file_SAx, Noneq_Spin_Ax);
    Write_Data(file_SBx, Noneq_Spin_Bx);
    file_C.close();
    file_Cs.close();
    file_SAx.close();
    file_SBx.close();
    


    
    
    /*
    cout << "Begin calculating the noneq_spin_sea phase diagram..."  << endl;
    VectorXf Param1 = linspace(-0.05*t, 0.05*t, N3);
    VectorXf Param2 = linspace(-0.1*t, 0.1*t, N4);
    MatrixXf Noneq_Spin_Ax(N3,N4), Noneq_Spin_Ay(N3,N4), Noneq_Spin_Az(N3,N4), Noneq_Spin_Bx(N3,N4), Noneq_Spin_By(N3,N4), Noneq_Spin_Bz(N3,N4);
    VectorXf temp(6);
    std::atomic<int> completedIterations(0);
    std::thread progressThread(trackProgress, std::ref(completedIterations), N3);
    #pragma omp parallel for private(KM, temp)
    for (int ii=0; ii<N3; ii++){
      KM.lso = Param1(ii);
      for (int jj=0; jj<N4; jj++){
        KM.lr = Param2(jj);
        temp = KM.Noneq_spin_sea(N1, N2, Gamma,0);
        Noneq_Spin_Ax(ii,jj) = temp(0);
        Noneq_Spin_Ay(ii,jj) = temp(1);
        Noneq_Spin_Az(ii,jj) = temp(2);
        Noneq_Spin_Bx(ii,jj) = temp(3);
        Noneq_Spin_By(ii,jj) = temp(4);
        Noneq_Spin_Bz(ii,jj) = temp(5);
      }
      completedIterations.fetch_add(1, std::memory_order_relaxed);
    }
    progressThread.join(); 
    cout << "\rProgress: " << 100 << "% (Fnished!)\n" << std::flush;
    cout << "Begin writing the data..."           << endl;
    cout << "----------------------------------"  << endl;
    const char *path_noneq_SAx="/Users/junyutang/Documents/cpp/projects/Kane_Mele_M/data/test/SAx(lso,lr).csv";
    //const char *path_noneq_SAy="/Users/junyutang/Documents/cpp/projects/Kane_Mele_M/data/noneq_spin_sea/SAy(lso,lr).csv";
    //const char *path_noneq_SAz="/Users/junyutang/Documents/cpp/projects/Kane_Mele_M/data/noneq_spin_sea/SAz(lso,lr).csv";
    const char *path_noneq_SBx="/Users/junyutang/Documents/cpp/projects/Kane_Mele_M/data/test/SBx(lso,lr).csv";
    //const char *path_noneq_SBy="/Users/junyutang/Documents/cpp/projects/Kane_Mele_M/data/noneq_spin_sea/SBy(lso,lr).csv";
    //const char *path_noneq_SBz="/Users/junyutang/Documents/cpp/projects/Kane_Mele_M/data/noneq_spin_sea/SBz(lso,lr).csv";
    ofstream file_noneq_SAx; file_noneq_SAx.open(path_noneq_SAx);
    //ofstream file_noneq_SAy; file_noneq_SAy.open(path_noneq_SAy);
    //ofstream file_noneq_SAz; file_noneq_SAz.open(path_noneq_SAz);
    ofstream file_noneq_SBx; file_noneq_SBx.open(path_noneq_SBx);
    //ofstream file_noneq_SBy; file_noneq_SBy.open(path_noneq_SBy);
    //ofstream file_noneq_SBz; file_noneq_SBz.open(path_noneq_SBz);
    //file_noneq_SAx << "noneq_S data file. lso: [-0.05,0.05] lr: [-0.1,0.1] lv=0.0 Jsd=0.1 E_ph =0. Gamma = 0.02 Row: lso Col: lr\n";
    //file_noneq_SAy << "noneq_S data file. lso: [-0.05,0.05] lr: [-0.1,0.1] lv=0.0 Jsd=0.1 E_ph =0. Gamma = 0.02 Row: lso Col: lr\n";
    //file_noneq_SAz << "noneq_S data file. lso: [-0.05,0.05] lr: [-0.1,0.1] lv=0.0 Jsd=0.1 E_ph =0. Gamma = 0.02 Row: lso Col: lr\n";
    //file_noneq_SBx << "noneq_S data file. lso: [-0.05,0.05] lr: [-0.1,0.1] lv=0.0 Jsd=0.1 E_ph =0. Gamma = 0.02 Row: lso Col: lr\n";
    //file_noneq_SBy << "noneq_S data file. lso: [-0.05,0.05] lr: [-0.1,0.1] lv=0.0 Jsd=0.1 E_ph =0. Gamma = 0.02 Row: lso Col: lr\n";
    //file_noneq_SBz << "noneq_S data file. lso: [-0.05,0.05] lr: [-0.1,0.1] lv=0.0 Jsd=0.1 E_ph =0. Gamma = 0.02 Row: lso Col: lr\n";
    Write_Data(file_noneq_SAx, Noneq_Spin_Ax);
    //Write_Data(file_noneq_SAy, Noneq_Spin_Ay);
    //Write_Data(file_noneq_SAz, Noneq_Spin_Az);
    Write_Data(file_noneq_SBx, Noneq_Spin_Bx);
    //Write_Data(file_noneq_SBy, Noneq_Spin_By);
    //Write_Data(file_noneq_SBz, Noneq_Spin_Bz);
    file_noneq_SAx.close();
    //file_noneq_SAy.close();
    //file_noneq_SAz.close();
    file_noneq_SBx.close();
    //file_noneq_SBy.close();
    //file_noneq_SBz.close();
    */
    
    
    /*
    cout << "Begin calculating the Ef dependence for noneq_spin_surf..." << endl;
    VectorXf Param1 = linspace(-1.2*t,1.2*t,601);
    MatrixXf neq_S(6,601);
    float Ef;
    std::atomic<int> completedIterations(0);
    std::thread progressThread(trackProgress, std::ref(completedIterations), N3);
    #pragma omp parallel for private(Ef)
    for (int ii=0; ii<601; ii++){
      Ef = Param1(ii);
      neq_S(all,ii) = KM.Noneq_spin_surface(N1, N2, Gamma, Ef, 0);
      completedIterations.fetch_add(1, std::memory_order_relaxed);
    }
    progressThread.join(); 
    cout << "\rProgress: " << 100 << "% (Fnished!)\n" << std::flush;
    cout << "Begin writing the data..."                      << endl;
    cout << "----------------------------------"             << endl;
    const char *path_neq_S="/Users/junyutang/Documents/cpp/projects/Kane_Mele_M/data/noneq_spin_surface/neq_S_vs_Ef.csv";
    ofstream file_neq_S; file_neq_S.open(path_neq_S);
    file_neq_S << "neq_S data file. Gamma=" << Gamma << "lso="<< KM.lso << "lv= "<< KM.lv << "lr= "<< KM.lr << "Jsd=" << KM.Jsd << " Row: Ax to Bz Col: Ef\n";
    Write_Data(file_neq_S, neq_S);
    file_neq_S.close();
    */
    
    /*
    cout << "Begin calculating the Chern number..."<< endl;
    VectorXf Param1 = linspace(-0.5*t, 0.5*t, N3);
    VectorXf Param2 = linspace(-1*t, 1*t, N4);
    MatrixXf Chern_number = MatrixXf::Zero(N3,N4);
    std::atomic<int> completedIterations(0);
    std::thread progressThread(trackProgress, std::ref(completedIterations), N3);
    #pragma omp parallel for private(KM)
    for (int ii=0; ii<N3; ii++){
      KM.lso = Param1(ii);
      for (int jj=0; jj<N4; jj++){
        KM.lv = Param2(jj);
        Chern_number(ii,jj) = KM.Chern_number_Feynman(N1,N2);
      }
      completedIterations.fetch_add(1, std::memory_order_relaxed);
    }
    progressThread.join(); 
    cout << "\rProgress: " << 100 << "% (Fnished!)\n" << std::flush;
    cout << "Begin writing the data..."            << endl;
    cout << "----------------------------------"   << endl;
    const char *path_Chern="/Users/junyutang/Documents/cpp/projects/Kane_Mele_M/data/test/testChern.csv";
    ofstream file_Chern; file_Chern.open(path_Chern);
    file_Chern << "Chern number data file. lso: [-0.05 0.05] lr: [-0.1 0.1] lv=0.05 lex=0.1. Row: soc Col: lr\n";
    Write_Data(file_Chern, Chern_number);
    file_Chern.close();
    */

    /*
    cout << "Begin calculating the Spin Chern number..."  << endl;
    VectorXf Param1 = linspace(-0.05,0.05,N3);
    VectorXf Param2 = linspace(-0.1,0.1,N4);
    MatrixXf Spin_Chern = MatrixXf::Zero(N3,N4);
    std::atomic<int> completedIterations(0);
    std::thread progressThread(trackProgress, std::ref(completedIterations), N3);
    #pragma omp parallel for private(KM)
    for (int ii=0; ii<N3; ii++){
      KM.lso  = Param1(ii);
      for (int jj=0; jj<N4; jj++){
        KM.lr = Param2[jj];
        Spin_Chern(ii,jj) = KM.Spin_Chern_number(N1,N2);
      }
      completedIterations.fetch_add(1, std::memory_order_relaxed);
    }
    progressThread.join(); 
    cout << "\rProgress: " << 100 << "% (Fnished!)\n" << std::flush;
    cout << "Begin writing the data..."           << endl;
    cout << "----------------------------------"  << endl;
    const char *path_sc = "/Users/junyutang/Documents/cpp/projects/Kane_Mele_M/data/testspinchern.csv";
    ofstream file_sc; file_sc.open(path_sc);
    file_sc << "Spin Chern number data file. lso: [-0.05:0.05] lr: [-0.1,0.1] lv=0.00 lex=0.1. Row: lso Col: lr\n";
    Write_Data(file_sc, Spin_Chern);
    file_sc.close();
    */

    /*
    cout << "Begin calculating the band bap phase..."  << endl;
    VectorXf Param1 = linspace(-0.05, 0.05, N3);
    VectorXf Param2 = linspace(-0.1,  0.1,  N4);
    MatrixXf Gap    = MatrixXf::Zero(N3,N4);
    std::atomic<int> completedIterations(0);
    std::thread progressThread(trackProgress, std::ref(completedIterations), N3);
    #pragma omp parallel for private(KM)
    for (int ii=0; ii<N3; ii++){
      KM.lso  = Param1(ii);
      for (int jj=0; jj<N4; jj++){
        KM.lr = Param2[jj];
        Gap(ii,jj) = KM.Get_global_bandgap(N1, N2);
      }
      completedIterations.fetch_add(1, std::memory_order_relaxed);
    }
    progressThread.join(); 
    cout << "\rProgress: " << 100 << "% (Fnished!)\n" << std::flush;
    cout << "Begin writing the data..."           << endl;
    cout << "----------------------------------"  << endl;
    const char *path_gap = "/Users/junyutang/Documents/cpp/projects/Kane_Mele_M/data/gap/lv=0.csv";
    Write_Data(path_gap, Gap);
    */


    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "\n Everything finished !" << endl;
    cout << "----------------------------------" << endl;
    cout << "Total time Taken : "<< duration.count()/1000000.0 << " seconds" << endl;
    cout << "----------------------------------" << endl;
    return 0;
}