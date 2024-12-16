//////////////////////
// QOI.cpp
// COPYRIGHT Yigong Qin, ALL RIGHTS RESERVED
/////////////////////

#include "QOI.h"
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <cstring>
#include <iostream>
using namespace std;

QOI::QOI()
: mNumActiveGrains(0)
{
}

QOILine::QOILine(const GlobalConstants params)
: mNumNodeFeatures(8)
{
    mQoIVectorFloatData.emplace("fractions", std::vector<float>((params.nts+1)*params.num_theta));

    mQoIVectorIntData.emplace("tip_y", std::vector<int>(params.nts+1));
    mQoIVectorIntData.emplace("cross_sec", std::vector<int>());
    mQoIVectorIntData.emplace("extra_area", std::vector<int>((params.nts+1)*params.num_theta, 0));
    mQoIVectorIntData.emplace("total_area", std::vector<int>((params.nts+1)*params.num_theta, 0));
    mQoIVectorIntData.emplace("tip_final", std::vector<int>((params.nts+1)*params.num_theta));

    // graph related QoIs
    int repeated_index = 5;
    mQoIVectorIntData.emplace("node_region", std::vector<int>((params.nts+1)*repeated_index*params.num_nodes*mNumNodeFeatures));
    std::fill(mQoIVectorIntData["node_region"].begin(), mQoIVectorIntData["node_region"].end(), -1);
}



void QOI::calculateLineQoIs(const GlobalConstants& params, int& cur_tip, const int* alpha, int kt, 
                        const float* z, const int* loss_area, int move_count)
{

     // cur_tip here inludes the halo
     int fnx = params.fnx, fny = params.fny, fnz = params.fnz, 
         num_grains = params.num_theta, all_time = params.nts+1;
     bool contin_flag = true;

     while(contin_flag == true)
     {
       // at this line
        cur_tip += 1;
        int offset_z = fnx*fny*cur_tip;
        //int zeros = 0;
        for (int j=1; j<fny-1; j++)
        {
            for (int i=1; i<fnx-1; i++)
            {
                int C = offset_z + j*fnx + i;
                //if (alpha[C]==0){printf("find liquid at %d at line %d\n", i, cur_tip);contin_flag=false;break;}
                if (alpha[C]==0) 
                {
                    contin_flag=false;
                    break;
                }
            }
        }
     }
     cur_tip -=1;
     mQoIVectorIntData["tip_y"][kt] = cur_tip + move_count; //z[cur_tip];
     printf("frame %d, ntip %d, tip %f\n", kt, mQoIVectorIntData["tip_y"][kt], z[cur_tip]);
     
     //cout << fnz << " " << fny << " " << fnx <<endl;
     for (int k = 1; k<fnz-1; k++)
     {
       int offset_z = fnx*fny*k; 
       for (int j = 1; j<fny-1; j++)
       { 
            for (int i = 1; i<fnx-1; i++)
            {
                int C = offset_z + fnx*j + i;
                if (alpha[C]>0)
                { 
                    //for (int time = kt; time<all_time; time++)
                    //{
                       // mQoIVectorIntData["tip_final"][time*num_grains+alpha[C]-1] = k+move_count;
                    //} 
                    mQoIVectorIntData["total_area"][kt*num_grains+alpha[C]-1]+=1;
                    if (k > cur_tip) 
                    {
                       mQoIVectorIntData["extra_area"][kt*num_grains+alpha[C]-1]+=1; 
                    }
                }
            }
       }
     }

     for (int j = 0; j<num_grains; j++)
     { 
         mQoIVectorIntData["total_area"][kt*num_grains+j]+=loss_area[j];
     }

}

void QOILine::searchJunctionsOnImage(const GlobalConstants& params, const int* alpha)
{
     // find the args that have active phs greater or equal 3, copy the args to mQoIVectorIntData["node_region"]
     int fnx = params.fnx, fny = params.fny;
   //  int start = (int) (params.z0/params.dx/params.W0);
  //   for (int cur_tip = start;  cur_tip<start + params.num_samples; cur_tip++){
     for (int kt = 0; kt<params.nts+1; kt++)
     {
        int cur_tip = mQoIVectorIntData["tip_y"][kt];
        int offset_z = fnx*fny*cur_tip;
        //memcpy(&mQoIVectorIntData["cross_sec"] + kt*fnx*fny, alpha + cur_tip*fnx*fny,  sizeof(int)*fnx*fny ); 
        copy(alpha + cur_tip*fnx*fny, alpha + (cur_tip + 1)*fnx*fny, back_inserter(mQoIVectorIntData["cross_sec"]));
        int offset_node_region = mQoIVectorIntData["node_region"].size()/(params.nts+1)*kt;
        int node_cnt = 0;
        for (int j = 1; j<fny-1; j++)
        { 
            for (int i = 1; i<fnx-1; i++)
            {
                int C = offset_z + j*fnx + i;
                unordered_map<int, int> occur;
                for (int dj = -1; dj<=1; dj++)
                { 
                    for (int di = -1; di<=1; di++)
                    {
                        int NC = offset_z + (j+dj)*fnx + i+di;
                        if (occur.find(alpha[NC])!=occur.end())
                        {
                            occur[alpha[NC]]++;
                        }
                        else
                        {
                            occur.insert({alpha[NC], 1});
                        }
                    }
                }       
                int alpha_occur=0, max_occur=0;
                for (auto & it : occur) 
                {
                    alpha_occur++;
                    max_occur = max(max_occur, it.second);
                }          
                if (alpha_occur>=3 && max_occur<=5)
                { 
                    mQoIVectorIntData["node_region"][offset_node_region + node_cnt*mNumNodeFeatures] = i;
                    mQoIVectorIntData["node_region"][offset_node_region + node_cnt*mNumNodeFeatures +1] = j;
                    mQoIVectorIntData["node_region"][offset_node_region + node_cnt*mNumNodeFeatures +2] = max_occur;
                    int pf_count = 0;
                    for (auto & it: occur)
                    {
                        mQoIVectorIntData["node_region"][offset_node_region + node_cnt*mNumNodeFeatures + 3 + pf_count] = it.first;
                        pf_count++;
                    } 
                    node_cnt++;
                }            
            }
        }
     }
}

void QOI::sampleHeights(int& cur_tip, const int* alpha, int fnx, int fny, int fnz)
{

     bool contin_flag = true;
     while(contin_flag == true)
     {
        cur_tip += 1;
        int offset_z = fnx*fny*cur_tip;

        for (int j=1; j<fny-1; j++)
        {
          for (int i=1; i<fnx-1; i++)
          {
             int C = offset_z + j*fnx + i;
             if (alpha[C]==0) 
             {
                contin_flag=false;
                break;
             }
          }
        }
     }
     cur_tip -=1;
}

QOI3D::QOI3D(const GlobalConstants params)
: mNumNodeFeatures(9)
{
    mQoIVectorIntData.emplace("volume", std::vector<int>());
    mQoIVectorIntData.emplace("grainToPF", std::vector<int>());
    mQoIVectorIntData.emplace("node_region", std::vector<int>());
    mQoIVectorIntData.emplace("manifold", std::vector<int>());
    mQoIVectorFloatData.emplace("geodesic_y_coors", std::vector<float>());
}


void QOI::calculateQoIs(const GlobalConstants& params, const int* alpha, int kt)
{
     mQoIVectorIntData["volume"].clear();
     mQoIVectorIntData["grainToPF"].clear();
     // cur_tip here inludes the halo
     int fnx = params.fnx, fny = params.fny, fnz = params.fnz, 
         num_grains = params.num_theta, all_time = params.nts+1;
     
     unordered_map<int, int> active_grains;
     mNumActiveGrains = 0;
     for (int k = 1; k<fnz-1; k++)
     {
       int offset_z = fnx*fny*k; 
       for (int j = 1; j<fny-1; j++)
       { 
            for (int i = 1; i<fnx-1; i++)
            {
                int C = offset_z + fnx*j + i;
                if (alpha[C]>0)
                {                   
                    if (active_grains.find(alpha[C])==active_grains.end())
                    {
                        active_grains.insert({alpha[C], mNumActiveGrains});
                        mNumActiveGrains++;
                        mQoIVectorIntData["grainToPF"].push_back(alpha[C]);
                        mQoIVectorIntData["volume"].push_back(0);
                    }
                    mQoIVectorIntData["volume"][active_grains[alpha[C]]] +=1;
                }
            }
       }
     }
    // mNumActiveGrains = active_grains.size();
}

void QOI3D::searchJunctionsOnImage(const GlobalConstants& params, const int* alpha)
{
     // find the args that have active phs greater or equal 3, copy the args to mQoIVectorIntData["node_region"]
     mQoIVectorIntData["node_region"].clear();
     mQoIVectorIntData["node_region"].resize(40*mNumActiveGrains*mNumNodeFeatures);
     std::fill(mQoIVectorIntData["node_region"].begin(), mQoIVectorIntData["node_region"].end(), -1);
     int fnx = params.fnx, fny = params.fny, fnz = params.fnz;
     int offset_node_region = 0;
     int node_cnt = 0;
     unordered_map<int, int> occur;

     for (int k = 1; k<fnz-1; k++)
     {
        for (int j = 1; j<fny-1; j++)
        { 
            for (int i = 1; i<fnx-1; i++)
            {
                int C = k*fnx*fny + j*fnx + i;
                if (alpha[C]==0) 
                {
                    continue;
                }
                occur.clear();
                int zeroOccur = 0;
                for (int dk = -1; dk<=1; dk++)
                {
                    for (int dj = -1; dj<=1; dj++)
                    { 
                        for (int di = -1; di<=1; di++)
                        {
                            if (di*di + dj*dj + dk*dk >2) 
                            {
                                continue;
                            }
                            int NC = (k+dk)*fnx*fny + (j+dj)*fnx + i+di;
                            if (alpha[NC]==0) 
                            {
                                zeroOccur++;
                            }
                            if (occur.find(alpha[NC])!=occur.end())
                            {
                                occur[alpha[NC]]++;
                            }
                            else
                            {
                                occur.insert({alpha[NC], 1});
                            }
                        }
                    }  
                }
                if (zeroOccur>0) 
                {
                    continue;
                }

                int alpha_occur=0, max_occur=0;
                for (auto & it : occur) 
                {
                    alpha_occur++;
                    max_occur = max(max_occur, it.second);
                }          
                if (alpha_occur==4 && max_occur<=10) // find a node
                { 
                    mQoIVectorIntData["node_region"][offset_node_region + node_cnt*mNumNodeFeatures] = i;
                    mQoIVectorIntData["node_region"][offset_node_region + node_cnt*mNumNodeFeatures +1] = j;
                    mQoIVectorIntData["node_region"][offset_node_region + node_cnt*mNumNodeFeatures +2] = k;
                    mQoIVectorIntData["node_region"][offset_node_region + node_cnt*mNumNodeFeatures +3] = max_occur;
                    int pf_count = 0;
                    for (auto & it: occur)
                    {
                        mQoIVectorIntData["node_region"][offset_node_region + node_cnt*mNumNodeFeatures + 4 + pf_count] = it.first;
                        pf_count++;
                    } 
                    node_cnt++;
                }            
            }
        }
     }
}


float cylindricalManifold(const GlobalConstants& params, float x, float y, float z, float t)
{
    float dist = sqrtf((y - 0.5f*params.lyd)*(y - 0.5f*params.lyd) + (z - params.z0 - params.lzd)*(z - params.z0 - params.lzd)) - params.r0;
                     
    return -params.G*dist - params.underCoolingRate*1e6*t;
}

void QOI::Manifold(const GlobalConstants& params, const int* alpha, const float* x, const float* y, const float* z, float t)
{
    int fnx = params.fnx, fny = params.fny, fnz = params.fnz;
    float prevTemperature, curTemperature;
    for (int i = 1; i<fnx-1; i++)
    {
        for (int j = 1; j<fny-1; j++)
        {
            prevTemperature = -10.0f;
            for (int k = 1; k<fnz-1; k++)
            {
                int C = k*fnx*fny + j*fnx + i;
                if (params.thermalType==2)
                {
                    curTemperature = cylindricalManifold(params, x[i], y[j], z[k], t);
                }
                if ( (curTemperature>0.0f) && (prevTemperature<0.0f) )
                {   
                    mQoIVectorIntData["manifold"].insert(mQoIVectorIntData["manifold"].end(), {i, j, k, alpha[C]});
                    if (params.thermalType==2) // cylindrical
                    {   
                        if (i==1)
                        {
                            mQoIVectorFloatData["geodesic_y_coors"].push_back(params.r0*asin( (y[j] - 0.5f*params.lyd)/params.r0 ));
                        }                       
                    }
                }
                prevTemperature = curTemperature;
            }
        }
    }

}
