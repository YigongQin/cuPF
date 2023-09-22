#include "params.h"
#include "PhaseField.h"
#include "helper.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <hdf5.h>
#include <random>
#include <assert.h>

#define LS -0.995

using namespace std;

void PhaseField::parseInputParams(std::string fileName)
{
    const DesignSettingData* designSetting = GetSetDesignSetting();

    float nts;
    float ictype;
    float haloWidth;
    float temp_Nx, temp_Ny, temp_Nz, temp_Nt;
    float nprd;
    ifstream parseFile(fileName);
    string lineText;
    while (parseFile.good()){
        std::getline (parseFile, lineText);
        // Output the text from the file
        //getParam(lineText, "G", params.G);
        //getParam(lineText, "R", params.R); 
        //getParam(lineText, "underCoolingRate", params.underCoolingRate); 
        getParam(lineText, "delta", params.delta); 
        getParam(lineText, "k", params.k); 
        //getParam(lineText, "c_infm", params.c_infm); 
        getParam(lineText, "Dh", params.Dh); 
        //getParam(lineText, "d0", params.d0); 
        getParam(lineText, "W0", params.W0);  
        getParam(lineText, "c_infty", params.c_infty);
        getParam(lineText, "m_slope", params.m_slope);
        //getParam(lineText, "beta0", params.beta0);
        getParam(lineText, "GT", params.GT);
        getParam(lineText, "L_cp", params.L_cp);
        getParam(lineText, "mu_k", params.mu_k);
        getParam(lineText, "eps", params.eps);
      //  getParam(lineText, "alpha0", params.alpha0);
        getParam(lineText, "dx", params.dx);
        getParam(lineText, "asp_ratio_yx", params.asp_ratio_yx);
        getParam(lineText, "asp_ratio_zx", params.asp_ratio_zx);
        getParam(lineText, "Lx", params.lxd);
      //  getParam(lineText, "nx", nx);
      //  params.nx = (int)nx;
      //  getParam(lineText, "Mt", Mt);
      //  params.Mt = (int)Mt;
        getParam(lineText, "eta", params.eta);
       // getParam(lineText, "U0", params.U0);
        getParam(lineText, "nts", nts);
        params.nts = (int)nts;
        getParam(lineText, "ictype", ictype);
        params.ictype = (int)ictype;
    //   getParam(lineText, "seed_val", seed_val);
    //    params.seed_val = (int)seed_val;
        getParam(lineText, "noi_period", nprd);
        params.noi_period = (int)nprd;
        getParam(lineText, "kin_delta", params.kin_delta);
        getParam(lineText, "beta0", params.beta0);
        getParam(lineText, "haloWidth", haloWidth);
        params.haloWidth = (int)haloWidth;
        getParam(lineText, "xmin", params.xmin);
        getParam(lineText, "ymin", params.ymin);
        getParam(lineText, "zmin", params.zmin);
      //  getParam(lineText, "num_theta", num_thetaf);
      //  params.num_theta = (int) num_thetaf;
        getParam(lineText, "nx", temp_Nx);
        getParam(lineText, "ny", temp_Ny);
        getParam(lineText, "nz", temp_Nz);
        getParam(lineText, "nt", temp_Nt);
        getParam(lineText, "cfl", params.cfl); 

        getParam(lineText, "Tmelt", params.Tmelt);
        getParam(lineText, "undcool_mean", params.undcool_mean);
        getParam(lineText, "undcool_std", params.undcool_std);
       // getParam(lineText, "nuc_Nmax", params.nuc_Nmax);
        getParam(lineText, "nuc_rad", params.nuc_rad);

        getParam(lineText, "moving_ratio", params.moving_ratio);
    }
    
    // Close the file
    parseFile.close();


    // macro input
    mac.Nx = (int) temp_Nx;
    mac.Ny = (int) temp_Ny;
    mac.Nz = (int) temp_Nz;
    mac.Nt = (int) temp_Nt;
    mac.X_mac = new float[mac.Nx];
    mac.Y_mac = new float[mac.Ny];
    mac.Z_mac = new float[mac.Nz];
    mac.t_mac = new float[mac.Nt];
    
    mac.psi_mac = new float [mac.Nx*mac.Ny*mac.Nz];
    mac.U_mac = new float [mac.Nx*mac.Ny*mac.Nz];
    mac.T_3D = new float[mac.Nx*mac.Ny*mac.Nz*mac.Nt];


    read_input(mac.folder+"/x.txt", mac.X_mac);
    read_input(mac.folder+"/y.txt", mac.Y_mac);
    read_input(mac.folder+"/z.txt", mac.Z_mac);
    read_input(mac.folder+"/t.txt", mac.t_mac);
   // read_input(mac.folder+"/alpha.txt",mac.alpha_mac);
    read_input(mac.folder+"/psi.txt",mac.psi_mac);
    read_input(mac.folder+"/U.txt",mac.U_mac);
    read_input(mac.folder+"/G.txt", &params.G);
    read_input(mac.folder+"/Rmax.txt", &params.R);
    read_input(mac.folder+"/NG.txt", &params.num_theta);
    read_input(mac.folder+"/NN.txt", &params.num_nodes);
    read_input(mac.folder+"/UC.txt", &params.underCoolingRate);
    read_input(mac.folder+"/Nmax.txt", &params.nuc_Nmax);
    read_input(mac.folder+"/z0.txt", &params.z0);
    read_input(mac.folder+"/top.txt", &params.top);

    mac.theta_arr = new float[2*params.num_theta+1];
    mac.cost = new float[2*params.num_theta+1];
    mac.sint = new float[2*params.num_theta+1];
    read_input(mac.folder+"/theta.txt", mac.theta_arr);

    hid_t  h5in_file,  datasetT, dataspaceT, memspace;
    hsize_t dimT[1];
    herr_t  status;
    dimT[0] = mac.Nx*mac.Ny*mac.Nz*mac.Nt; 
    h5in_file = H5Fopen( (mac.folder+"/Temp.h5").c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    datasetT = H5Dopen2(h5in_file, "Temp", H5P_DEFAULT);
    dataspaceT = H5Dget_space(datasetT);
    memspace = H5Screate_simple(1,dimT,NULL);
    status = H5Dread(datasetT, H5T_NATIVE_FLOAT, memspace, dataspaceT,
                     H5P_DEFAULT, mac.T_3D);
    printf("mac.T %f\n",mac.T_3D[mac.Nx*mac.Ny*mac.Nz*mac.Nt-1]); 
    H5Dclose(datasetT);
    H5Sclose(dataspaceT);
    H5Sclose(memspace);
    H5Fclose(h5in_file);
    
    if (designSetting->includeNucleation)
    {
        float domainScaleFactor = cbrt(0.01/params.nuc_Nmax);
        params.lxd = 2*(params.lxd*domainScaleFactor/2);
    }

 

    float dxd = params.dx*params.W0;
    params.c_infm = params.c_infty*params.m_slope;
    params.Tliq = params.Tmelt - params.c_infm;
    params.Tsol = params.Tmelt - params.c_infm/params.k;
    params.Ti = params.Tsol;   
    params.d0 = params.GT/params.L_cp;
    params.beta0 = 1.0/(params.mu_k*params.L_cp);
    //params.lT = params.c_infm*( 1.0/params.k-1 )/params.G;//       # thermal length           um
    params.lamd = 0.8839*params.W0/params.d0;//     # coupling constant
    //params.tau0 = 0.6267*params.lamd*params.W0*params.W0/params.Dl; //    # time scale  
    params.tau0 = params.beta0*params.lamd*params.W0/0.8839;
    //params.kin_coeff = tauk/params.tau0;
    //params.R_tilde = params.R*params.tau0/params.W0;
    //params.Dl_tilde = params.Dl*params.tau0/pow(params.W0,2);
    //params.lT_tilde = params.lT/params.W0;
    params.beta0_tilde = params.beta0*params.W0/params.tau0;
    params.dt = params.cfl*params.dx*params.beta0_tilde;
//    params.ny = (int) (params.asp_ratio*params.nx);
//    params.lyd = params.asp_ratio*params.lxd;
    params.hi = 1.0/params.dx;
    params.cosa = cos(params.alpha0/180*M_PI);
    params.sina = sin(params.alpha0/180*M_PI);
    params.sqrt2 = sqrt(2.0);
    params.a_s = 1 - 3.0*params.delta;
    params.epsilon = 4.0*params.delta/params.a_s;
    params.a_12 = 4.0*params.a_s*params.epsilon;
    params.dt_sqrt = sqrt(params.dt);

    params.nx = (int) round(params.lxd/params.dx/params.W0);//global cells 
    params.ny = (int) round(params.asp_ratio_yx*params.nx);
    if (designSetting->useLineConfig == true)
    {
        params.nz = (int) round(params.moving_ratio*params.nx);
    }
    else
    {
        params.nz = (int) round(params.asp_ratio_zx*params.nx);
    }
    params.nz_full = (int) round(params.asp_ratio_zx*params.nx);
    params.lxd = params.nx*dxd;
    params.lyd = params.ny*dxd;
    params.lzd = params.nz*dxd;
    params.Mt = (int) (mac.t_mac[mac.Nt-1]/params.tau0/params.dt);
    params.Mt = (params.Mt/2)*2; 
    int kts = params.Mt/params.nts;
    kts = (kts/2)*2;
    params.Mt = kts*params.nts;
    params.tmax = params.tau0*params.dt*params.Mt;

    params.pts_cell = (int) (params.nuc_rad/dxd);

    params.bcX = designSetting->bcX;
    params.bcY = designSetting->bcY;
    params.bcZ = designSetting->bcZ;

    if (designSetting->useLineConfig || params.underCoolingRate>0.0)
    {
        params.thermalType = 1;
    }
    else
    {
        params.thermalType = 0;
    }

    if (GetMPIManager()->rank==0)
    {
        string thermalType = params.thermalType ==1 ? "analytic" : "interpolation";
        string bcX = params.bcX == 0 ? "noflux" : "periodic";
        string bcY = params.bcY == 0 ? "noflux" : "periodic";
        string bcZ = params.bcZ == 0 ? "noflux" : "periodic";
        cout << "thermalType: " << thermalType << endl;
        cout << "bcX: " << bcX << endl;
        cout << "bcY: " << bcY << endl;
        cout << "bcZ: " << bcZ << endl;
    }

}

void PhaseField::cpuSetup(MPIsetting* mpiManager){

    if (mpiManager->rank==0)
    { 
    // print params and mac information

    cout<<"dx = "<<params.lxd/params.nx/params.W0<<endl;
    cout<<"dy = "<<params.lyd/params.ny/params.W0<<endl;   
    cout<<"dz = "<<params.lzd/params.nz/params.W0<<endl;  
    cout<<"nx = "<<params.nx<<endl;
    cout<<"ny = "<<params.ny<<endl;
    cout<<"nz = "<<params.nz<<endl;

    cout<<"noise coeff = "<<params.dt_sqrt*params.hi*params.eta<<endl;


    cout<<"mac Nx = "<<mac.Nx<<endl;
    cout<<"mac Ny = "<<mac.Ny<<endl;
    cout<<"mac Nz = "<<mac.Nz<<endl;
    cout<<"mac Nt = "<<mac.Nt<<endl;
    cout<<"macro dimension = "<<mac.Nx*mac.Ny*mac.Nz<<endl;

    cout<<"number of grains = "<<params.num_theta<<endl;
    cout<<"number of time steps = "<<params.Mt<<endl;

    }

    mpiManager->nxLocal = params.nx/mpiManager->numProcessorX;
    mpiManager->nyLocal = params.ny/mpiManager->numProcessorY;
    mpiManager->nzLocal = params.nz/mpiManager->numProcessorZ;
    mpiManager->nzLocalAll = params.nz_full;
    mpiManager->haloWidth = params.haloWidth;
    
    float dxd = params.dx*params.W0;
    float len_blockx = mpiManager->nxLocal*dxd; 
    float len_blocky = mpiManager->nyLocal*dxd; 
    float len_blockz = mpiManager->nzLocal*dxd; 
    
    float xmin_loc = params.xmin + mpiManager->processorIDX*len_blockx; 
    float ymin_loc = params.ymin + mpiManager->processorIDY*len_blocky; 
    float zmin_loc = params.zmin + mpiManager->processorIDZ*len_blockz;

    if (mpiManager->processorIDX == 0)
    {
        mpiManager->nxLocal += 1;
    }
    else
    {
        xmin_loc += dxd;
    } 
    
    if (mpiManager->processorIDY == 0)
    {
        mpiManager->nyLocal += 1;
    }
    else
    {
        ymin_loc += dxd;
    }

    mpiManager->nzLocal += 1;
    mpiManager->nzLocalAll += 1;

    params.fnx = mpiManager->nxLocal + 2*params.haloWidth;
    params.fny = mpiManager->nyLocal + 2*params.haloWidth;
    params.fnz = mpiManager->nzLocal + 2*params.haloWidth;
    params.fnz_f = mpiManager->nzLocalAll + 2*params.haloWidth;

    fnx = params.fnx;
    fny = params.fny;
    fnz = params.fnz;
    fnz_f = params.fnz_f;

    params.length = fnx*fny*fnz;
    params.full_length = fnx*fny*fnz_f;
    length = params.length;
    full_length = params.full_length;

    x = new float[fnx];
    y = new float[fny];
    z = new float[fnz];
    z_full = new float[fnz_f];

    for(int i=0; i<fnx; i++){
        x[i]=(i-params.haloWidth)*dxd + xmin_loc; 
    }

    for(int i=0; i<fny; i++){
        y[i]=(i-params.haloWidth)*dxd + ymin_loc;
    }

    for(int i=0; i<fnz; i++){
        z[i]=(i-params.haloWidth)*dxd + zmin_loc;
    }

    for(int i=0; i<fnz_f; i++){
        z_full[i]=(i-params.haloWidth)*dxd + zmin_loc;
    }


    psi = new float[length];
    phi = new float[length];
  //  Uc = new float[length];
    alpha = new int[length];
    alpha_i_full = new int[full_length];

  //  cout<<"x= ";
  //  for(int i=0; i<fnx; i++){
  //      cout<<x[i]<<" ";
  //  }

    cout<< "rank "<< mpiManager->rank<< " xmin " << x[0] << " xmax "<<x[fnx-1]<<endl;
    cout<< "rank "<< mpiManager->rank<< " ymin " << y[0] << " ymax "<<y[fny-1]<<endl;
    cout<< "rank "<< mpiManager->rank<< " zmin " << z[0] << " zmax "<<z[fnz-1]<<endl;
    cout<<"x length of psi, phi, U="<<fnx<<endl;
    cout<<"y length of psi, phi, U="<<fny<<endl;
    cout<<"z length of psi, phi, U="<<fnz<<endl;   
    cout<<"length of psi, phi, U="<<length<<endl;
}


void PhaseField::initField(){

    for (int i=0; i<2*params.num_theta+1; i++){
        //mac.theta_arr[i+1] = 1.0f*(rand()%10)/(10-1)*(-M_PI/2.0);
       // mac.theta_arr[i+1] = 1.0f*rand()/RAND_MAX*(-M_PI/2.0);
       // mac.theta_arr[i+1] = (i)*grain_gap- M_PI/2.0;
        mac.sint[i] = sinf(mac.theta_arr[i]);
        mac.cost[i] = cosf(mac.theta_arr[i]);
    }  
    mac.alpha_mac = new int [(params.nx+1)*(params.ny+1)];
    read_input(mac.folder+"/alpha.txt", mac.alpha_mac);
cout << mac.alpha_mac[0] << " " << mac.alpha_mac[(params.nx+1)*(params.ny+1)-1] <<endl;
    //printf("%d %d\n", mac.alpha_mac[0], mac.alpha_mac[(fnx-2*params.haloWidth)*(fny-2*params.haloWidth)-1]);
     
    float Dx = mac.X_mac[mac.Nx-1] - mac.X_mac[mac.Nx-2];
    float Dy = mac.Y_mac[mac.Ny-1] - mac.Y_mac[mac.Ny-2];
    float Dz = mac.Z_mac[mac.Nz-1] - mac.Z_mac[mac.Nz-2];    
    for(int id=0; id<length; id++)
    {
      int k = id/(fnx*fny);
      int k_r = id - k*fnx*fny;
      int j = k_r/fnx;
      int i = k_r%fnx; 
      
      if ( (i>params.haloWidth-1) && (i<fnx-params.haloWidth) && (j>params.haloWidth-1) && (j<fny-params.haloWidth) && (k>params.haloWidth-1) && (k<fnz-params.haloWidth))
      {
        int kx = (int) (( x[i] - mac.X_mac[0] )/Dx);
        float delta_x = ( x[i] - mac.X_mac[0] )/Dx - kx;
            //printf("%f ",delta_x);
        int ky = (int) (( y[j] - mac.Y_mac[0] )/Dy);
        float delta_y = ( y[j] - mac.Y_mac[0] )/Dy - ky;

        int kz = (int) (( z[k] - mac.Z_mac[0] )/Dz);
        float delta_z = ( z[k] - mac.Z_mac[0] )/Dz - kz;

        if (kx==mac.Nx-1) {kx = mac.Nx-2; delta_x =1.0f;}
        if (ky==mac.Ny-1) {ky = mac.Ny-2; delta_y =1.0f;}
        if (kz==mac.Nz-1) {kz = mac.Nz-2; delta_z =1.0f;}

        int offset =  kx + ky*mac.Nx + kz*mac.Nx*mac.Ny;
        int offset_n =  kx + ky*mac.Nx + (kz+1)*mac.Nx*mac.Ny;
        //if (offset>mac.Nx*mac.Ny-1-1-mac.Nx) printf("%d, %d, %d, %d  ", i,j,kx,ky);
        psi[id] = ( (1.0f-delta_x)*(1.0f-delta_y)*mac.psi_mac[ offset ] + (1.0f-delta_x)*delta_y*mac.psi_mac[ offset+mac.Nx ] \
                +delta_x*(1.0f-delta_y)*mac.psi_mac[ offset+1 ] +   delta_x*delta_y*mac.psi_mac[ offset+mac.Nx+1 ] )*(1.0f-delta_z) + \
                ( (1.0f-delta_x)*(1.0f-delta_y)*mac.psi_mac[ offset_n ] + (1.0f-delta_x)*delta_y*mac.psi_mac[ offset_n+mac.Nx ] \
                +delta_x*(1.0f-delta_y)*mac.psi_mac[ offset_n+1 ] +   delta_x*delta_y*mac.psi_mac[ offset_n+mac.Nx+1 ] )*delta_z;

        psi[id] = psi[id]/params.W0;
        phi[id]=tanhf(psi[id]/params.sqrt2);

        if (phi[id]>LS && GetSetDesignSetting()->pureNucleation == false)
        {
            alpha[id] = mac.alpha_mac[(j-1)*(fnx-2*params.haloWidth)+(i-1)];
            if (alpha[id]<1 || alpha[id]>params.num_theta) cout<<"found alpha out of bounds at "<<(j-1)*(fnx-2*params.haloWidth)+(i-1)<< " alpha "<<alpha[id]<<endl;
        }
        else 
        {
            alpha[id]=0;
        }
      }
      else
      {
        psi[id]=0.0f;
        phi[id]=0.0f;
        alpha[id]=0;
      }
    }



}

std::string to_stringp(float a_value, int n );
void h5write_1d(hid_t h5_file, const char* name, void* data, int length, std::string dtype);

void PhaseField::OutputQoIs()
{
    const DesignSettingData* designSetting = GetSetDesignSetting(); 
    if (designSetting->useLineConfig)
    {
        qois->searchJunctionsOnImage(params, alpha_i_full);
    }
    else
    {
        qois->searchJunctionsOnImage(params, alpha);
    }

    string grainType, outputFormat;
    if (designSetting->includeNucleation)
    {
        grainType = "Mixed_density" + to_string(params.nuc_Nmax) + "_grains" + to_string(qois->mNumActiveGrains);
        outputFormat = grainType +"_nodes"+to_string(params.num_nodes)+"_frames"+to_string(params.nts)+"_lxd"+to_string(params.lxd)+\
                          "_G"+to_stringp(params.G,3)+"_UC"+to_stringp(params.underCoolingRate,3)+"_seed"+to_string(params.seed_val)+"_Mt"+to_string(params.Mt);
    }
    else
    {
        grainType = "Epita_grains"+to_string(params.num_theta);
        outputFormat = grainType +"_nodes"+to_string(params.num_nodes)+"_frames"+to_string(params.nts)+"_lxd"+to_string(params.lxd)+\
                          "_G"+to_stringp(params.G,3)+"_Rmax"+to_stringp(params.R,3)+"_seed"+to_string(params.seed_val)+"_Mt"+to_string(params.Mt);
    }   
    string outputFile = outputFormat+ "_rank"+to_string(GetMPIManager()->rank)+".h5";

    outputFile = designSetting->outputFolder + '/' + outputFile;
    cout << "save file name: " << outputFile << endl;

    hid_t  h5_file; 

    h5_file = H5Fcreate(outputFile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    h5write_1d(h5_file, "x_coordinates", x, fnx, "float");
    h5write_1d(h5_file, "y_coordinates", y, fny, "float");
    h5write_1d(h5_file, "z_coordinates", z_full, params.fnz_f, "float");
    h5write_1d(h5_file, "angles",    mac.theta_arr, (2*params.num_theta+1), "float");

    for (auto & qoi : qois->mQoIVectorFloatData)
    {
        h5write_1d(h5_file, qoi.first.c_str(), qoi.second.data(), qoi.second.size(), "float");
    }

    for (auto & qoi : qois->mQoIVectorIntData)
    {
        h5write_1d(h5_file, qoi.first.c_str(), qoi.second.data(), qoi.second.size(), "int");
    }

    H5Fclose(h5_file);
}

void PhaseField::OutputField(int currentStep)
{
    const DesignSettingData* designSetting = GetSetDesignSetting(); 
    string grainType, outputFormat;
    if (designSetting->includeNucleation)
    {
        grainType = "Mixed_density" + to_string(params.nuc_Nmax) + "_grains" + to_string(qois->mNumActiveGrains);
        outputFormat = grainType +"_nodes"+to_string(params.num_nodes)+"_frames"+to_string(params.nts)+"_lxd"+to_string(params.lxd)+\
                          "_G"+to_stringp(params.G,3)+"_UC"+to_stringp(params.underCoolingRate,3)+"_seed"+to_string(params.seed_val)+"_Mt"+to_string(params.Mt);
    }
    else
    {
        grainType = "Epita_grains"+to_string(params.num_theta);
        outputFormat = grainType +"_nodes"+to_string(params.num_nodes)+"_frames"+to_string(params.nts)+"_lxd"+to_string(params.lxd)+\
                          "_G"+to_stringp(params.G,3)+"_Rmax"+to_stringp(params.R,3)+"_seed"+to_string(params.seed_val)+"_Mt"+to_string(params.Mt);
    }
    string outputFile = outputFormat+ "_rank"+to_string(GetMPIManager()->rank) + "_time" + to_string(currentStep) + ".h5";

    outputFile = designSetting->outputFolder + '/' + outputFile;
    cout << "save file name " << outputFile << endl;

    hid_t  h5_file; 

    h5_file = H5Fcreate(outputFile.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    h5write_1d(h5_file, "x_coordinates", x, fnx, "float");
    h5write_1d(h5_file, "y_coordinates", y, fny, "float");
    h5write_1d(h5_file, "z_coordinates", z_full, params.fnz_f, "float");
    h5write_1d(h5_file, "angles",    mac.theta_arr, (2*params.num_theta+1), "float");

    if (designSetting->useLineConfig)
    {
        h5write_1d(h5_file, "alpha",  alpha_i_full, full_length, "int");
    }
    else{
        h5write_1d(h5_file, "alpha",  alpha, length, "int");
    }

    H5Fclose(h5_file);

    string cmd = "python3 saveVTKdata.py --rawdat_dir=" + designSetting->outputFolder + " --rank=" + to_string(GetMPIManager()->rank) + " --seed=" + to_string(params.seed_val)
                    + " --lxd=" + to_string((int) params.lxd) + " --time=" + to_string(currentStep);
    std::cout << cmd << std::endl;
    int result = system(cmd.c_str()); 
    assert(result == 0);
}


std::string to_stringp(float a_value, int n )
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

void h5write_1d(hid_t h5_file, const char* name, void* data, int length, std::string dtype){

    herr_t  status;
    hid_t dataspace, h5data=0;
    hsize_t dim[1];
    dim[0] = length;
    
    dataspace = H5Screate_simple(1, dim, NULL);

    if (dtype.compare("int") ==0){

        h5data = H5Dcreate2(h5_file, name, H5T_NATIVE_INT, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(h5data, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    }
    else if (dtype.compare("float") ==0)
    {
        h5data = H5Dcreate2(h5_file, name, H5T_NATIVE_FLOAT_g, dataspace,H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        status = H5Dwrite(h5data, H5T_NATIVE_FLOAT_g, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    }
    else 
    {
        printf("the data type not specifed");
        status = 1;
    }

    H5Sclose(dataspace);
    H5Dclose(h5data);
}
