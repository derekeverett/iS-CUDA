/*
Matthew Golden
June 2017
*/

#ifndef __CUDA__EMISSION__INCLUDED__
#define __CUDA__EMISSION__INCLUDED__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include "emissionfunction.h" //Includes main.h ParameterReader.h Table.h

#define threadsPerBlock 512
//The number of blocks needed is calculated, not defined
#define debug 0				//1 for debugging, 0 otherwise

/*
This is a header file that defines the function calculate_dN_ptdptdphidy_GPU() for the EmissionFunctionArray

First, all needed information is arranged into arrays, then device copies are made.

The first  kernel performs the Cooper-Frye integral in parallel.
Each thread is assigned a cell of the freezeout surface.
The thread then loops over all particles and momenta.

After all threads have computed a set contributions to a certain momenta and mass, it sums over threads.
These partial sums are stored in an array dN_pTdpTdphidy[numofparticle * numofmomenta * numofblocks]
numofblocks is the number of blocks used to contain the freezeout surface

The next kernel reduction then does the secondary sum over the block dimension to yield a final complete spectrum
This is then copied back to the host and written to a file.
*/

__device__ void getbulkvisCoefficients(double Tdec, double* bulkvisCoefficients, double hbarC, int bulk_deltaf_kind)
{
   double Tdec_fm = Tdec/hbarC;  // [1/fm]
   double Tdec_fm_power[11];    // cache the polynomial power of Tdec_fm
   Tdec_fm_power[1] = Tdec_fm;
   for(int ipower = 2; ipower < 11; ipower++)
       Tdec_fm_power[ipower] = Tdec_fm_power[ipower-1]*Tdec_fm;
   /*if(bulk_deltaf_kind == 0)       // 14 moment expansion
   {
        // load from file
        bulkvisCoefficients[0] = bulkdf_coeff->interp(1, 2, Tdec_fm, 5)/pow(hbarC, 3);  //B0 [fm^3/GeV^3]
        bulkvisCoefficients[1] = bulkdf_coeff->interp(1, 3, Tdec_fm, 5)/pow(hbarC, 2);  // D0 [fm^3/GeV^2]
        bulkvisCoefficients[2] = bulkdf_coeff->interp(1, 4, Tdec_fm, 5)/pow(hbarC, 3);  // E0 [fm^3/GeV^3]
        // parameterization for mu = 0
        //bulkvisCoefficients[0] = exp(-15.04512474*Tdec_fm + 11.76194266)/pow(hbarC, 3); //B0[fm^3/GeV^3]
        //bulkvisCoefficients[1] = exp( -12.45699277*Tdec_fm + 11.4949293)/hbarC/hbarC;  // D0 [fm^3/GeV^2]
        //bulkvisCoefficients[2] = -exp(-14.45087586*Tdec_fm + 11.62716548)/pow(hbarC, 3);  // E0 [fm^3/GeV^3]
   }*/
   // use else if if previous code block is uncommented
   if(bulk_deltaf_kind == 1)  // relaxation type
   {
       // parameterization from JF
       // A Polynomial fit to each coefficient -- X is the temperature in fm^-1
       // Both fits are reliable between T=100 -- 180 MeV , do not trust it beyond
       bulkvisCoefficients[0] = (  642096.624265727
                                 - 8163329.49562861*Tdec_fm_power[1]
                                 + 47162768.4292073*Tdec_fm_power[2]
                                 - 162590040.002683*Tdec_fm_power[3]
                                 + 369637951.096896*Tdec_fm_power[4]
                                 - 578181331.809836*Tdec_fm_power[5]
                                 + 629434830.225675*Tdec_fm_power[6]
                                 - 470493661.096657*Tdec_fm_power[7]
                                 + 230936465.421*Tdec_fm_power[8]
                                 - 67175218.4629078*Tdec_fm_power[9]
                                 + 8789472.32652964*Tdec_fm_power[10]);

       bulkvisCoefficients[1] = (  1.18171174036192
                                 - 17.6740645873717*Tdec_fm_power[1]
                                 + 136.298469057177*Tdec_fm_power[2]
                                 - 635.999435106846*Tdec_fm_power[3]
                                 + 1918.77100633321*Tdec_fm_power[4]
                                 - 3836.32258307711*Tdec_fm_power[5]
                                 + 5136.35746882372*Tdec_fm_power[6]
                                 - 4566.22991441914*Tdec_fm_power[7]
                                 + 2593.45375240886*Tdec_fm_power[8]
                                 - 853.908199724349*Tdec_fm_power[9]
                                 + 124.260460450113*Tdec_fm_power[10]);
   }
   else if (bulk_deltaf_kind == 2)
   {
       // A Polynomial fit to each coefficient -- Tfm is the temperature in fm^-1
       // Both fits are reliable between T=100 -- 180 MeV , do not trust it beyond
       bulkvisCoefficients[0] = (
               21091365.1182649 - 290482229.281782*Tdec_fm_power[1]
             + 1800423055.01882*Tdec_fm_power[2] - 6608608560.99887*Tdec_fm_power[3]
             + 15900800422.7138*Tdec_fm_power[4] - 26194517161.8205*Tdec_fm_power[5]
             + 29912485360.2916*Tdec_fm_power[6] - 23375101221.2855*Tdec_fm_power[7]
             + 11960898238.0134*Tdec_fm_power[8] - 3618358144.18576*Tdec_fm_power[9]
             + 491369134.205902*Tdec_fm_power[10]);

       bulkvisCoefficients[1] = (
               4007863.29316896 - 55199395.3534188*Tdec_fm_power[1]
             + 342115196.396492*Tdec_fm_power[2] - 1255681487.77798*Tdec_fm_power[3]
             + 3021026280.08401*Tdec_fm_power[4] - 4976331606.85766*Tdec_fm_power[5]
             + 5682163732.74188*Tdec_fm_power[6] - 4439937810.57449*Tdec_fm_power[7]
             + 2271692965.05568*Tdec_fm_power[8] - 687164038.128814*Tdec_fm_power[9]
             + 93308348.3137008*Tdec_fm_power[10]);
   }
   else if (bulk_deltaf_kind == 3)
   {
       bulkvisCoefficients[0] = (
               160421664.93603 - 2212807124.97991*Tdec_fm_power[1]
             + 13707913981.1425*Tdec_fm_power[2] - 50204536518.1767*Tdec_fm_power[3]
             + 120354649094.362*Tdec_fm_power[4] - 197298426823.223*Tdec_fm_power[5]
             + 223953760788.288*Tdec_fm_power[6] - 173790947240.829*Tdec_fm_power[7]
             + 88231322888.0423*Tdec_fm_power[8] - 26461154892.6963*Tdec_fm_power[9]
             + 3559805050.19592*Tdec_fm_power[10]);
       bulkvisCoefficients[1] = (
               33369186.2536556 - 460293490.420478*Tdec_fm_power[1]
             + 2851449676.09981*Tdec_fm_power[2] - 10443297927.601*Tdec_fm_power[3]
             + 25035517099.7809*Tdec_fm_power[4] - 41040777943.4963*Tdec_fm_power[5]
             + 5682163732.74188*Tdec_fm_power[6] - 4439937810.57449*Tdec_fm_power[7]
             + 2271692965.05568*Tdec_fm_power[8] - 687164038.128814*Tdec_fm_power[9]
             + 93308348.3137008*Tdec_fm_power[10]);
   }
   else if (bulk_deltaf_kind == 3)
   {
       bulkvisCoefficients[0] = (
               160421664.93603 - 2212807124.97991*Tdec_fm_power[1]
             + 13707913981.1425*Tdec_fm_power[2] - 50204536518.1767*Tdec_fm_power[3]
             + 120354649094.362*Tdec_fm_power[4] - 197298426823.223*Tdec_fm_power[5]
             + 223953760788.288*Tdec_fm_power[6] - 173790947240.829*Tdec_fm_power[7]
             + 88231322888.0423*Tdec_fm_power[8] - 26461154892.6963*Tdec_fm_power[9]
             + 3559805050.19592*Tdec_fm_power[10]);
       bulkvisCoefficients[1] = (
               33369186.2536556 - 460293490.420478*Tdec_fm_power[1]
             + 2851449676.09981*Tdec_fm_power[2] - 10443297927.601*Tdec_fm_power[3]
             + 25035517099.7809*Tdec_fm_power[4] - 41040777943.4963*Tdec_fm_power[5]
             + 46585225878.8723*Tdec_fm_power[6] - 36150531001.3718*Tdec_fm_power[7]
             + 18353035766.9323*Tdec_fm_power[8] - 5504165325.05431*Tdec_fm_power[9]
             + 740468257.784873*Tdec_fm_power[10]);
   }
   else if (bulk_deltaf_kind == 4)
   {
       bulkvisCoefficients[0] = (
               1167272041.90731 - 16378866444.6842*Tdec_fm_power[1]
             + 103037615761.617*Tdec_fm_power[2] - 382670727905.111*Tdec_fm_power[3]
             + 929111866739.436*Tdec_fm_power[4] - 1540948583116.54*Tdec_fm_power[5]
             + 1767975890298.1*Tdec_fm_power[6] - 1385606389545*Tdec_fm_power[7]
             + 709922576963.213*Tdec_fm_power[8] - 214726945096.326*Tdec_fm_power[9]
             + 29116298091.9219*Tdec_fm_power[10]);
       bulkvisCoefficients[1] = (
               5103633637.7213 - 71612903872.8163*Tdec_fm_power[1]
             + 450509014334.964*Tdec_fm_power[2] - 1673143669281.46*Tdec_fm_power[3]
             + 4062340452589.89*Tdec_fm_power[4] - 6737468792456.4*Tdec_fm_power[5]
             + 7730102407679.65*Tdec_fm_power[6] - 6058276038129.83*Tdec_fm_power[7]
             + 3103990764357.81*Tdec_fm_power[8] - 938850005883.612*Tdec_fm_power[9]
             + 127305171097.249*Tdec_fm_power[10]);
   }
   return;
}

__global__ void cooperFrye( long FO_length, int number_of_chosen_particles, int pT_tab_length, int phi_tab_length, int eta_tab_length, double* dN_pTdpTdphidy_d, double* pT_d, double* trig_d, double* hyperTrig_d, double* delta_eta_d, double* mass_d, double* sign_d, double* degen_d, int* baryon_d, double* gammaT_d, double* ux_d, double* uy_d, double* mu_d, double* muB_d, double* Tdec_d, double* Edec_d, double* Pdec_d, double* da0_d, double* da1_d, double* da2_d, double* tau_d, double prefactor, double* pi00_d, double* pi01_d, double* pi02_d, double* pi11_d, double* pi12_d, double* pi22_d, double* pi33_d, double* bulkPi_d, double hbarC, int bulk_deltaf_kind, int INCLUDE_DELTAF, int INCLUDE_BULKDELTAF, int F0_IS_NOT_SMALL)
{
        //This array is a shared array that will contain the integration contributions from each cell.
        __shared__ double temp[threadsPerBlock];

        //Assign a global index and a local index
        int idx      = threadIdx.x + blockDim.x*blockIdx.x;
        int localIdx = threadIdx.x;
        __syncthreads();

	//assuming bulkdeltafcoef doesn't depnd on mass, move into loop if needed
	/*double *bulkvisCoefficients;
        if(bulk_deltaf_kind == 0)
        	bulkvisCoefficients = new double [3];
        else
               	bulkvisCoefficients = new double [2];
        //I'm just going to make it three and not use the last one if i dont need it
        */
	//Declare things that do not depend on momentum outside of loop
        double bulkvisCoefficients[3] = {0.,0.,0.};
        if(INCLUDE_BULKDELTAF == 1)
        {
        	//if(bulk_deltaf_kind == 0)
                //      bulkPi_d[idx] = bulkPi_d[idx];
                if(bulk_deltaf_kind != 0)
                	bulkPi_d[idx] = bulkPi_d[idx]/hbarC;   // unit in fm^-4
                getbulkvisCoefficients(Tdec_d[idx], bulkvisCoefficients, hbarC, bulk_deltaf_kind);
        }
	double deltaf_prefactor = 1.0/( 2.0*Tdec_d[idx]*Tdec_d[idx]*(Edec_d[idx] + Pdec_d[idx]) );


        for( int mmidx = 0; mmidx < number_of_chosen_particles * pT_tab_length * phi_tab_length; mmidx++ )
       	{
        	temp[localIdx] = 0;
                //If thread is outside the range of the surface, skip calculations and leaves temp[localIdx]=0
                if(idx<FO_length){
                //Count over pT, then phi, then particle
                int pTIdx       = mmidx % pT_tab_length;
                int phiIdx      = ( mmidx/pT_tab_length)%phi_tab_length;
                int partIdx     = mmidx / (pT_tab_length*phi_tab_length);
                double px       = pT_d[pTIdx]*trig_d[phiIdx+phi_tab_length];
                double py       = pT_d[pTIdx]*trig_d[phiIdx];
                double mT       = sqrt(mass_d[partIdx]*mass_d[partIdx]+pT_d[pTIdx]*pT_d[pTIdx]);
                for(int k=0; k<eta_tab_length; k++)
               {
                	double pt = mT*hyperTrig_d[k+eta_tab_length];
                        double pz = mT*hyperTrig_d[k];

                        double pdotu = pt*gammaT_d[idx] - px*ux_d[idx] - py*uy_d[idx];

                        double expon = ( pdotu - mu_d[idx*number_of_chosen_particles + partIdx] - baryon_d[partIdx]*muB_d[idx] ) / Tdec_d[idx];
                        double f0      = 1./(exp(expon)+sign_d[partIdx]);
                        double pdsigma = pt*da0_d[idx] + px*da1_d[idx] + py*da2_d[idx];
                        double delta_f_shear = 0.0;
                        double delta_f_bulk = 0.0;

                        if(INCLUDE_DELTAF)
                  	{
                      		double Wfactor = (
                          		  pt*pt*pi00_d[idx] - 2.0*pt*px*pi01_d[idx] - 2.0*pt*py*pi02_d[idx]
                        		+ px*px*pi11_d[idx] + 2.0*px*py*pi12_d[idx] + py*py*pi22_d[idx]
                           		+ pz*pz*pi33_d[idx]);
                      		delta_f_shear = ((1 - F0_IS_NOT_SMALL*sign_d[partIdx]*f0)
                                       *Wfactor*deltaf_prefactor);
                  	}

			if(INCLUDE_BULKDELTAF == 1)
			{
         			if(bulk_deltaf_kind == 0)
                          		delta_f_bulk = (
                             			 - (1. - F0_IS_NOT_SMALL*sign_d[partIdx]*f0)*bulkPi_d[idx]
                               			 *(bulkvisCoefficients[0]*mass_d[partIdx]*mass_d[partIdx]
                                  		 + bulkvisCoefficients[1]*pdotu
                                  		 + bulkvisCoefficients[2]*pdotu*pdotu));
                      		else if (bulk_deltaf_kind == 1)
                      		{
                          		double E_over_T = pdotu/Tdec_d[idx];
                          		double mass_over_T = mass_d[partIdx]/Tdec_d[idx];
                          		delta_f_bulk = (
                              			-1.0*(1.-sign_d[partIdx]*f0)/E_over_T*bulkvisCoefficients[0]
                              			*(mass_over_T*mass_over_T/3.
                              		  	- bulkvisCoefficients[1]*E_over_T*E_over_T)
                              			*bulkPi_d[idx]);
                      		}
                      		else if (bulk_deltaf_kind == 2)
                      		{
                          		double E_over_T = pdotu/Tdec_d[idx];
                          		delta_f_bulk = (-1.*(1.-sign_d[partIdx]*f0)
                                       		   *(-bulkvisCoefficients[0]
                                        	    + bulkvisCoefficients[1]*E_over_T)
                                        	   *bulkPi_d[idx]);
                      		}
				else if (bulk_deltaf_kind == 3)
                      		{
                          		double E_over_T = pdotu/Tdec_d[idx];
                          		delta_f_bulk = (-1.0*(1.-sign_d[partIdx]*f0)/sqrt(E_over_T)
                                          		*(-bulkvisCoefficients[0]
                                            		+ bulkvisCoefficients[1]*E_over_T)
                                          		*bulkPi_d[idx]);
                      		}
                     		else if (bulk_deltaf_kind == 4)
                      		{
                          		double E_over_T = pdotu/Tdec_d[idx];
                          		delta_f_bulk = (-1.0*(1.-sign_d[partIdx]*f0)
                                          	*(bulkvisCoefficients[0]
                                            	- bulkvisCoefficients[1]/E_over_T)
                                          	*bulkPi_d[idx]);
                      		}
			}


			//Insert statements calculating delta_f here, ignoring it for testing
                        double ratio = min(1., fabs( 1./(delta_f_shear + delta_f_bulk) ));
                        double result = prefactor*degen_d[partIdx]*pdsigma*tau_d[idx]*f0*(1. + (delta_f_shear + delta_f_bulk)*ratio);
                        temp[localIdx] += result*delta_eta_d[k];
                }}//finish if(idx<FO_length)

                int N = blockDim.x;
                __syncthreads(); //Make sure threads are prepared for reduction

                do
                {
			//Note, N must be a power of 2
                        N /= 2;
                        if(localIdx < N)        temp[localIdx] += temp[localIdx + N];
                        __syncthreads();         //Test if this is needed
                }while(N!=1);

                int spectra_size = number_of_chosen_particles*pT_tab_length*phi_tab_length;
                if(localIdx == 0)
			dN_pTdpTdphidy_d[blockIdx.x*spectra_size +  mmidx] = temp[0];
	}
}

//Does a block sum, where the previous kernel did a thread sum.
__global__ void reduction(double* dN_pTdpTdphidy_d, int final_spectrum_size, int cooperfryeblocks)
{
        int idx = threadIdx.x + blockDim.x*blockIdx.x;
        if(idx<final_spectrum_size)
        {
                if(cooperfryeblocks == 1) return; //Probably will never happen, but best to be careful
                //Need to start at i=1, since adding everything to i=0
                for(int i = 1; i < cooperfryeblocks; i++)
                        dN_pTdpTdphidy_d[idx] += dN_pTdpTdphidy_d[idx + i*final_spectrum_size];
        }
}


void EmissionFunctionArray::calculate_dN_ptdptdphidy_GPU()
{
  cudaDeviceSynchronize();
  cudaError_t err;

  //caclulate all important numbers to the problem first to report them
  long spectrum_size = ( (FO_length + threadsPerBlock - 1)/threadsPerBlock ) * number_of_chosen_particles * pT_tab_length * phi_tab_length; //the size of the spectrum which has intrablock reduced, but not interblock reduced
  int cooperfryeblocks = (FO_length + threadsPerBlock - 1)/threadsPerBlock; //the total number of blocks needed
  int final_spectrum_size = number_of_chosen_particles*pT_tab_length*phi_tab_length; //size of final array for all particles , as a function of particle, pT and phi
  int blocks = (final_spectrum_size + threadsPerBlock -1)/threadsPerBlock;

  cout << "Important Numbers:" << endl;
  cout << "# of chosen particles   = " << number_of_chosen_particles << endl;
  cout << "FO_length               = " << FO_length      << endl;
  cout << "pT_tab_length           = " << pT_tab_length  << endl;
  cout << "phi_tab_length          = " << phi_tab_length << endl;
  cout << "eta_tab_length          = " << eta_tab_length << endl;
  cout << "unreduced spectrum size = " << spectrum_size  << endl;
  cout << "reduced spectrum size   = " << final_spectrum_size  << endl;
  cout << "threads per block       = " << threadsPerBlock<< endl;
  cout << "blocks in first kernel  = " << cooperfryeblocks<< endl;
  cout << "blocks in second kernel = " << blocks  << endl;

  //Convert object data into arrays the GPU can more easily work with
  //Particle Properties
  cout << endl << "starting to declare host arrays" <<endl;
  double  mass[number_of_chosen_particles];
  double  sign[number_of_chosen_particles];
  double  degen[number_of_chosen_particles];
  int     baryon[number_of_chosen_particles];

  //Momentum properties
  double  pT[pT_tab_length];
  double  trig[2*phi_tab_length];         //Contains sin then cos for discrete phi, no sense in calculating them 50,000 times
  double  hyperTrig[2*eta_tab_length];    //Same as above, but cosh and sinh
  double  delta_eta[eta_tab_length];      //These are the weights for quadrature

  //Freeze out surface properties
  double Tdec[FO_length];
  double Pdec[FO_length];
  double Edec[FO_length];

  //mu will enumerate mu for all particles before increasing to next cell
  double *mu;
  mu = (double*) malloc( FO_length*number_of_chosen_particles*sizeof(double) );
  double tau[FO_length];
  double gammaT[FO_length];
  double ux[FO_length];
  double uy[FO_length];
  double da0[FO_length];
  double da1[FO_length];
  double da2[FO_length];
  double pi00[FO_length];
  double pi01[FO_length];
  double pi02[FO_length];
  double pi11[FO_length];
  double pi12[FO_length];
  double pi22[FO_length];
  double pi33[FO_length];
  double muB[FO_length];
  double bulkPi[FO_length];
  double *dN_pTdpTdphidy;
  cout << "declaring dN_pTdpTdphidy, hope it doesn't crash" << endl;
  dN_pTdpTdphidy = (double*) malloc( spectrum_size*sizeof(double) );

  cout << "declared arrays" <<endl;

  //Fill arrays with data
  for(int i=0; i<spectrum_size; i++) dN_pTdpTdphidy[i] = 0.0;
  for(int i=0; i<number_of_chosen_particles; i++)
  {
    int            particle_idx  = chosen_particles_sampling_table[i];
    particle_info *particle      = &particles[particle_idx];

    mass[i]   = particle->mass  ;
    sign[i]   = particle->sign  ;
    degen[i]  = particle->gspin ;
    baryon[i] = particle->baryon;
  }
  for(int i=0;              i<pT_tab_length;  i++) pT[i]        = pT_tab->get(1, i+1);
  for(int i=0;              i<phi_tab_length; i++) trig[i]      = sin( phi_tab->get(1, i+1) );
  for(int i=0;              i<phi_tab_length; i++) trig[i+phi_tab_length] = cos( phi_tab->get(1, i+1) );

  //I do not know what particle_y is, but the original code uses it
  for(int i=0;              i<eta_tab_length; i++) hyperTrig[i] = sinh(particle_y - eta_tab->get(1, i+1) );
  for(int i=0;              i<eta_tab_length; i++) hyperTrig[i+eta_tab_length] = cosh( particle_y - eta_tab->get(1, i+1) );
  for(int i=0;              i<eta_tab_length; i++) delta_eta[i] = eta_tab->get(2, i+1);

  for(int i=0; i<FO_length; i++)
  {
    FO_surf *surf = &FOsurf_ptr[i];

    Tdec[i] = surf->Tdec;
    Pdec[i] = surf->Pdec;
    Edec[i] = surf->Edec;
    for(int j=0; j<number_of_chosen_particles;j++)
    {
      int particle_idx  = chosen_particles_sampling_table[j];
      mu[i*number_of_chosen_particles + j] = surf->particle_mu[particle_idx];
    }
    tau[i]   = surf->tau;
    gammaT[i]= surf->u0;
    ux[i]    = surf->u1;
    uy[i]    = surf->u2;
    da0[i]   = surf->da0;
    da1[i]   = surf->da1;
    da2[i]   = surf->da2;
    pi00[i]  = surf->pi00;
    pi01[i]  = surf->pi01;
    pi02[i]  = surf->pi02;
    pi11[i]  = surf->pi11;
    pi12[i]  = surf->pi12;
    pi22[i]  = surf->pi22;
    pi33[i]  = surf->pi33;
    muB[i]   = surf->muB;
    bulkPi[i]= surf->bulkPi;
  }

  cout << "declaring device variables" << endl;
  //Make device copies of all of these arrays
  double *mass_d;
  double *sign_d;
  double *degen_d;
  int    *baryon_d;

  //Momentum properties
  double *pT_d;
  double *trig_d;
  double *hyperTrig_d;
  double *delta_eta_d;

  //Freeze out surface properties
  double *Tdec_d;
  double *Pdec_d;
  double *Edec_d;
  double *mu_d;
  double *tau_d;
  double *gammaT_d;
  double *ux_d;
  double *uy_d;
  double *da0_d;
  double *da1_d;
  double *da2_d;
  double *pi00_d;
  double *pi01_d;
  double *pi02_d;
  double *pi11_d;
  double *pi12_d;
  double *pi22_d;
  double *pi33_d;
  double *muB_d;
  double *bulkPi_d;
  double *dN_pTdpTdphidy_d;

  cout<< "allocating memory for device variables" << endl;
  //Allocate a lot of memory on device
  cudaMalloc( (void**) &mass_d,   number_of_chosen_particles*sizeof(double)       );
  cudaMalloc( (void**) &sign_d,   number_of_chosen_particles*sizeof(double)       );
  cudaMalloc( (void**) &degen_d,  number_of_chosen_particles*sizeof(double)       );
  cudaMalloc( (void**) &baryon_d, number_of_chosen_particles*sizeof(int)          );
  cudaMalloc( (void**) &pT_d,     pT_tab_length*sizeof(double)                    );
  cudaMalloc( (void**) &trig_d,   2*phi_tab_length*sizeof(double)                 );
  cudaMalloc( (void**) &hyperTrig_d,   2*eta_tab_length*sizeof(double)            );
  cudaMalloc( (void**) &delta_eta_d, eta_tab_length*sizeof(double)                );
  cudaMalloc( (void**) &Tdec_d,   FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &Pdec_d,   FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &Edec_d,   FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &mu_d,     number_of_chosen_particles*FO_length*sizeof(double)  );
  cudaMalloc( (void**) &tau_d,    FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &gammaT_d, FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &ux_d,     FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &uy_d,     FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &da0_d,    FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &da1_d,    FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &da2_d,    FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &pi00_d,   FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &pi01_d,   FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &pi02_d,   FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &pi11_d,   FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &pi12_d,   FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &pi22_d,   FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &pi33_d,   FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &muB_d,    FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &bulkPi_d, FO_length*sizeof(double)                        );
  cudaMalloc( (void**) &dN_pTdpTdphidy_d, spectrum_size*sizeof(double)            );

  cout << "finished allocating" << endl;
  cout << "starting to copy data to device" << endl;

  //Copy the CPU variables to GPU
  cudaMemcpy( mass_d,   	mass,   number_of_chosen_particles*sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( sign_d,   	sign,   number_of_chosen_particles*sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( degen_d,  	degen,  number_of_chosen_particles*sizeof(double),   cudaMemcpyHostToDevice );
  cudaMemcpy( baryon_d, 	baryon, number_of_chosen_particles*sizeof(int),      cudaMemcpyHostToDevice );
  cudaMemcpy( pT_d,     	pT,     pT_tab_length*sizeof(double),                cudaMemcpyHostToDevice );
  cudaMemcpy( trig_d,   	trig,   2*phi_tab_length*sizeof(double),             cudaMemcpyHostToDevice );
  cudaMemcpy( hyperTrig_d,hyperTrig,  2*eta_tab_length*sizeof(double),         cudaMemcpyHostToDevice );
  cudaMemcpy( delta_eta_d,delta_eta, eta_tab_length*sizeof(double),            cudaMemcpyHostToDevice );
  cudaMemcpy( Tdec_d,   	Tdec,   FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( Pdec_d,   	Pdec,   FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( Edec_d,   	Edec,   FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( mu_d,     	mu,     number_of_chosen_particles*FO_length*sizeof(double), cudaMemcpyHostToDevice );
  cudaMemcpy( tau_d,    	tau,    FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( gammaT_d, 	gammaT, FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( ux_d,     	ux,     FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( uy_d,     	uy,     FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( da0_d,    	da0,    FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( da1_d,    	da1,    FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( da2_d,    	da2,    FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( pi00_d,   	pi00,   FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( pi01_d,   	pi01,   FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( pi02_d,   	pi02,   FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( pi11_d,   	pi11,   FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( pi12_d,   	pi12,   FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( pi22_d,   	pi22,   FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( pi33_d,   	pi33,   FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( muB_d,   	muB,    FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( bulkPi_d,   bulkPi, FO_length*sizeof(double),                    cudaMemcpyHostToDevice );
  cudaMemcpy( dN_pTdpTdphidy_d, dN_pTdpTdphidy, spectrum_size*sizeof(double),  cudaMemcpyHostToDevice );

  cout << "Finished copying memory." << endl;

  //Perform kernels, first inital cooper Frye and reduction acriss threads, second is another reduction across blocks
  double prefactor = 1.0/(8.0*(M_PI*M_PI*M_PI))/hbarC/hbarC/hbarC;

  if(debug) cout << "Starting first cooper-frye kernel" << endl;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  cudaDeviceSynchronize();

  cooperFrye<<<cooperfryeblocks,threadsPerBlock>>>( FO_length, number_of_chosen_particles, pT_tab_length, phi_tab_length, eta_tab_length, dN_pTdpTdphidy_d, pT_d, trig_d, hyperTrig_d, delta_eta_d, mass_d, sign_d, degen_d, baryon_d, gammaT_d, ux_d, uy_d, mu_d, muB_d, Tdec_d, Edec_d, Pdec_d, da0_d, da1_d, da2_d, tau_d, prefactor, pi00_d, pi01_d, pi02_d, pi11_d, pi12_d, pi22_d, pi33_d,  bulkPi_d, hbarC, bulk_deltaf_kind, INCLUDE_DELTAF, INCLUDE_BULKDELTAF, F0_IS_NOT_SMALL);
  cudaDeviceSynchronize();

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("Error in first kernel: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;
  }

  cout << "finished first kernel" << endl;

  cout << "starting the second kernel" << endl;

  cudaDeviceSynchronize();
  reduction<<<blocks,threadsPerBlock>>>(dN_pTdpTdphidy_d, final_spectrum_size, cooperfryeblocks);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cout << "This took " << milliseconds << " milliseconds." << endl;

  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("Error in second kernel: %s\n", cudaGetErrorString(err));
    err = cudaSuccess;
  }

  cout << "finished second kernel" << endl;

  //Copy spectra back to host
  cout << "Copy data back to host" << endl;

  cudaMemcpy( dN_pTdpTdphidy, dN_pTdpTdphidy_d, number_of_chosen_particles*pT_tab_length*phi_tab_length*sizeof(double),  cudaMemcpyDeviceToHost );

  cout << "finished copying data back to host" << endl;
  cout << "Writing spectra to files" << endl;

  //Write results to files

  ofstream of1(dN_ptdptdphidy_filename.c_str(), ios_base::app);

  //Follow the old convention for storing data
  for(int i=0; i<Nparticles; i++)
  {
    //The bouncer. See if a particle is chosen.
    for(int counter = 0; counter< number_of_chosen_particles; counter++)
    {
      if(chosen_particles_sampling_table[counter] == i)
      {
        for (int phiIdx=0; phiIdx<phi_tab_length; phiIdx++)
        {
          for (int pTIdx=0; pTIdx<pT_tab_length; pTIdx++)
          {
            of1 << scientific <<  setw(15) << setprecision(8) << dN_pTdpTdphidy[pTIdx + pT_tab_length*phiIdx + counter*pT_tab_length*phi_tab_length] << "  ";
          }
          of1 << endl; //Only new lines for new angles
        }
        break;
      }
      //If counter reaches max value, particle wasnt chosen, output zeros
      if(counter == number_of_chosen_particles - 1)
      {
        for (int phiIdx=0; phiIdx<phi_tab_length; phiIdx++)
        {
          for (int pTIdx=0; pTIdx<pT_tab_length; pTIdx++)
          {
            of1 << scientific <<  setw(15) << setprecision(8) << 0.  << "  ";
          }
          of1 << endl; //Only new lines for new angles
        }
      }
    }
  }
  of1.close();


  cout << "finished writing spectra to files" << endl;
  cout << "Freeing device memory and cpu memory" << endl;

  //Free Memory
  free( mu );
  free( dN_pTdpTdphidy );
  cudaFree( mass_d );
  cudaFree( sign_d );
  cudaFree( degen_d );
  cudaFree( baryon_d );
  cudaFree( pT_d );
  cudaFree( trig_d );
  cudaFree( hyperTrig_d );
  cudaFree( delta_eta_d );
  cudaFree( Tdec_d );
  cudaFree( Pdec_d );
  cudaFree( Edec_d );
  cudaFree( mu_d );
  cudaFree( tau_d );
  cudaFree( gammaT_d );
  cudaFree( ux_d );
  cudaFree( uy_d );
  cudaFree( da0_d );
  cudaFree( da1_d );
  cudaFree( da2_d );
  cudaFree( pi00_d );
  cudaFree( pi01_d );
  cudaFree( pi02_d );
  cudaFree( pi11_d );
  cudaFree( pi12_d );
  cudaFree( pi22_d );
  cudaFree( pi33_d );
  cudaFree( muB_d  );
  cudaFree( bulkPi_d );
  cudaFree( dN_pTdpTdphidy_d );
  cout << "Finished everything! Have a good day!" << endl;
}


#endif
