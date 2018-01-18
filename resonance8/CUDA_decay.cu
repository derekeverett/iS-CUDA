/* 
This code was rewritten with cuda to run on a GPU. The structure is mostly the same as on the CPU code, 
but there are parallelizations over decays and momentum space.
*/

#include <cuda.h>
#include <cuda_runtime.h>

#ifndef __CUDA__DECAY__
#define __CUDA__DECAY__

//This can be changed for debugging purposes. Look at locations in code to see what will turn on and off.
#define DEBUG 100

#define PTS3 12  /* normalization of 3-body decay */
#define PTS4 12  /* inv. mass integral 3-body */

#define PTN1 8
#define PTN2 8  /* 2-body between the poles */

//#include "decay.cu" switched so now decay.cu includes this code
#include "CUDA_int.cu" // just int.cpp but a CUDA file and everything is a device function
#include "tools.cu"   // just with all functions as device functions

#define PTCHANGE        1.0
#define threadsPerBlock 256

#ifndef __NBLOCKDEF__
#define __NBLOCKDEF__
__device__ typedef struct nblock
{
  double a, b, c, d;
}nblock;                        /* for normalisation integral of 3-body decays */
#endif

#ifndef __PBLOCKDEF__
#define __PBLOCKDEF__
__device__ typedef struct pblockN
{
  double pt, mt, y, e, pl;      // pt, mt, y of decay product 1 
  double phi;
  double m1, m2, m3;            // masses of decay products     
  double mr;                    // mass of resonance            
  double costh, sinth;
  double e0, p0;
  int res_num;                  // Montecarlo number of the Res. 
}pblockN;
#endif

//Change to a exponentially based interpolation for better results
__device__ double  Edndp3(double yr, double ptr, double phirin, int res_num, int *partid_d, int npt_max, int nphi_max, double *ptarray_d, double *phiarray_d, double* dN_ptdptdphidy_d)
// double   yr;     /* y  of resonance */
// double   ptr;        /* pt of resonance */
// double   phirin;     /* phi angle  of resonance */
// int  res_num;    /* Montecarlo number of resonance   */
{
    if(DEBUG <= 9) return 1.0;
    double phir, val;
    double f1, f2;
    int pn, npt, nphi;

    //Make sure in bounds
    /*if(phirin < 0.0)
    {
        printf("ERROR: phir %15.8le < 0 !!! \n", phirin);
        exit(0);
    }
    if(phirin > 2.0*PI)
    {
        printf("ERROR: phir %15.8le > 2PI !!! \n", phirin);
        exit(0);
    }*/
    phir = phirin;
    pn = partid_d[MHALF + res_num];
    npt = 1;

    //while((ptr > particle[pn].pt[npt]) && (npt<(particle[pn].npt - 1)))
    while( ( ptr > ptarray_d[npt] ) && ( npt < (npt_max - 1) ) )
        npt++;

    //take as an argument of function
    //int nphi_max = particle[pn].nphi-1;
    //note arrays go from 0 to var_max-1 for some variable
    if(phir < phiarray_d[0])
    {
	//return 1.0;
        /*f1 = lin_int(PHI[nphi_max]-2*M_PI, PHI[0],
                     particle[pn].dNdptdphi[npt-1][particle[pn].nphi-1],
                     particle[pn].dNdptdphi[npt-1][0], phir);
        f2 = lin_int(PHI[nphi_max]-2*M_PI, PHI[0],
                     particle[pn].dNdptdphi[npt][nphi_max],
                     particle[pn].dNdptdphi[npt][0], phir);*/

        //The index algebra is not nice. 
  	f1 = lin_int(phiarray_d[nphi_max-1]-2.0*M_PI, phiarray_d[0],
		     dN_ptdptdphidy_d[pn*npt_max*nphi_max + npt_max*(nphi_max-1) + npt-1],
	 	     dN_ptdptdphidy_d[pn*npt_max*nphi_max + 0 + npt-1], phir);
        f2 = lin_int(phiarray_d[nphi_max-1]-2.0*M_PI, phiarray_d[0],
		     dN_ptdptdphidy_d[pn*npt_max*nphi_max + npt_max*(nphi_max-1) + npt],
		     dN_ptdptdphidy_d[pn*npt_max*nphi_max + 0 + npt], phir);
    }
    //phiarray_d has nphi_max elements
    else if( phir > phiarray_d[nphi_max-1] )
    {
	//return 1.0;
        f1 = lin_int(phiarray_d[nphi_max-1]+2.0*M_PI, phiarray_d[0],
                     dN_ptdptdphidy_d[pn*npt_max*nphi_max + npt_max*(nphi_max-1) + npt-1],
                     dN_ptdptdphidy_d[pn*npt_max*nphi_max + 0 + npt-1], phir);
        f2 = lin_int(phiarray_d[nphi_max-1]+2.0*M_PI, phiarray_d[0],
                     dN_ptdptdphidy_d[pn*npt_max*nphi_max + npt_max*(nphi_max-1) + npt],
                     dN_ptdptdphidy_d[pn*npt_max*nphi_max + 0 + npt], phir);
    }
    else
    {
	//return 1.0;
        nphi = 1;
        while( ( phir > phiarray_d[nphi] ) && ( nphi < (nphi_max-1) ) ) nphi++;
        /* phi interpolation */
        f1 = lin_int(phiarray_d[nphi-1], phiarray_d[nphi],
 		     dN_ptdptdphidy_d[pn*npt_max*nphi_max + npt_max*(nphi-1) + npt-1],
                     dN_ptdptdphidy_d[pn*npt_max*nphi_max + npt_max*nphi + npt-1], phir);
        f2 = lin_int(phiarray_d[nphi-1], phiarray_d[nphi],
		     dN_ptdptdphidy_d[pn*npt_max*nphi_max + npt_max*(nphi-1) + npt],
                     dN_ptdptdphidy_d[pn*npt_max*nphi_max + npt_max*nphi + npt], phir);
    }
    if(isnan(f1)) printf("f1 is NAN for phir = %f \n", phir);
    if(isnan(f2)) printf("f2 is NAN for phir = %f \n", phir);
    if(f1 < 0) f1 = 0.0;
    if(f2 < 0) f2 = 0.0;
    f1 = f1 + 1e-100;
    f2 = f2 + 1e-100;
    if(ptr > PTCHANGE)
    {
           f1 = logf(f1);
           f2 = logf(f2);
    }
    val = lin_int(ptarray_d[npt-1], ptarray_d[npt], f1, f2, ptr);
    if(isnan(val))
    {
	//printf("lin_int has failed for val\n");
        /*printf("dEdndy3 val \n");
        printf("f1, %15.8lf f2, %15.8lf  \n", f1, f2);
        printf("nphi  %i npt %i \n", nphi,npt);
        printf("f1  %15.8le %15.8le \n", f1, f2);
        printf("phi  %15.8lf %15.8lf \n", PHI[nphi-1], PHI[nphi]);
        printf("pt   %15.8lf %15.8lf \n", particle[pn].pt[npt-1],particle[pn].pt[npt]);
        printf("phi  %15.8lf, pt %15.8lf, val %15.8lf \n", phir, ptr,val);
        printf("phi %15.8le %15.8le \n", particle[pn].dNdptdphi[npt][nphi-1],
                                         particle[pn].dNdptdphi[npt][nphi]);
        printf("pt  %15.8le %15.8le \n", particle[pn].dNdptdphi[npt-1][nphi-1],
                                         particle[pn].dNdptdphi[npt-1][nphi]);
        exit(-1);*/
    }
    if(ptr > PTCHANGE)
        val = exp(val);
    //printf(" nphi  %i npt %i \n", nphi,npt);
    //printf(" f1  %15.8le %15.8le  \n", f1, f2);
    //printf(" phi  %15.8lf %15.8lf  \n", PHI[nphi-1], PHI[nphi]);
    //printf(" pt   %15.8lf %15.8lf  \n", particle[pn].pt[npt-1],particle[pn].pt[npt]);
    //printf(" phi  %15.8lf pt %15.8lf    val %15.8lf \n", phir, ptr,val);
    //printf(" phi %15.8le %15.8le \n",particle[pn].dNdptdphi[npt][nphi-1],
    //                            particle[pn].dNdptdphi[npt][nphi]);
    //printf(" pt  %15.8le %15.8le \n",particle[pn].dNdptdphi[npt-1][nphi-1],
    //                            particle[pn].dNdptdphi[npt-1][nphi]);

    //exit(0);
    return val;
}

//Only arguments that matter are up to *paranorm. All others are dummy variables for the gauss function.
__device__ double norm3int (double x, void *paranorm, int *partid_d, int npt_max, int nphi_max, double *ptarray_d, double *phiarray_d, double *dN_ptdptdphidy)
{
  if(DEBUG <= 4) return 1.0;
  nblock *tmp = (nblock *) paranorm;
  double res = sqrt((tmp->a - x)*(tmp->b - x)
                        *(x - tmp->c)*(x - tmp->d))/x;
  //printf("norm3int = %f for a = %f, b = %f, c = %f, d = %f, and x = %f\n", res, tmp->a, tmp->b, tmp->c, tmp->d, x);
  return res;
}

__device__ double dnpir2N (double phi, void *para1, int *partid_d, int npt_max, int nphi_max, double *ptarray_d, double *phiarray_d, double* dN_ptdptdphidy_d)
{
    if(DEBUG <= 8) return 1.0;
    pblockN *para = (pblockN *) para1;
    double D;
    double eR, plR, ptR, yR, phiR, sume, jac;
    double cphiR, sphiR;
    double dnr;                 /* dn/mtdmt of resonance */

    sume = para->e + para->e0;

    D = para->e * para->e0 + para->pl * para->p0 * para->costh +
        para->pt * para->p0 * para->sinth * cos (phi) + para->m1 * para->m1;

    eR = para->mr * (sume * sume / D - 1.0);
    jac = para->mr + eR;
    plR = para->mr * sume * (para->pl - para->p0 * para->costh) / D;
    ptR = (eR * eR - plR * plR - para->mr * para->mr);

    if(ptR < 0.0)
        ptR = 0.0;
    else
        ptR = sqrt(ptR);

    yR = 0.5 * logf ((eR + plR) / (eR - plR));
    cphiR = - jac * (para->p0 * para->sinth * cos (phi + para->phi)
            - para->pt * cos (para->phi)) / (sume * ptR);
    sphiR = - jac * (para->p0 * para->sinth * sin (phi + para->phi)
            - para->pt * sin (para->phi)) / (sume * ptR);
    if ((fabs (cphiR) > 1.000) || (fabs (sphiR) > 1.000))
    {
       /*if ((fabs (cphiR) > 1.0001) || (fabs (sphiR) > 1.0001))
        {
            //printf ("  |phir| = %15.8lf  > 1 ! \n", phiR);
            printf (" phi %15.8le D %15.8le \n", phi, D);
            printf (" eR %15.8le plR %15.8le \n", eR, plR);
            printf (" ptR %15.8le jac %15.8le \n", ptR, jac);
            printf (" sume %15.8le costh %15.8le \n", sume, para->costh);

            printf (" pt %15.8le \n", para->pt);
            printf (" mt  %15.8le \n", para->mt);
            printf (" y %15.8le \n", para->y);
            printf (" e %15.8le \n", para->e);
            printf (" e0 %15.8le \n", para->e0);
            printf (" p0 %15.8le \n", para->p0);
            printf (" pl %15.8le \n", para->pl);
            printf (" phi %15.8le \n", para->phi);

            printf (" m1 %15.8le \n", para->m1);
            printf (" m2 %15.8le \n", para->m2);
            printf (" m3 %15.8le \n", para->m3);
            printf (" mr %15.8le \n", para->mr);
            exit (0);
        }*/
        //else
        //{
            if (cphiR > 1.0)
                cphiR = 1.0;
            if (cphiR < -1.0)
                cphiR = -1.0;
        //}
    }

    phiR = acos (cphiR);
    if (sphiR < 0.0)
        phiR = 2.0 * PI - phiR;
    //if(isnan(yR)) printf("dNpir2N yR \n");
    //if(isnan(ptR)) printf("dNpir2N ptR \n");
    //if(isnan(phiR)) printf("dNpir2N phiR \n");
    //if(isnan(para->res_num)) printf("dNpir2N res_num \n");

    dnr = Edndp3 (yR, ptR, phiR, para->res_num, partid_d, npt_max, nphi_max, ptarray_d, phiarray_d, dN_ptdptdphidy_d);
 
   //printf(" phir = %15.8lf  ! ", phiR);
    //printf(" ptR %15.8le jac %15.8le ", ptR, jac );
    //printf(" dnr %15.8le \n", dnr); 
    return dnr * jac * jac / (2.0 * sume * sume);
}

__device__ double dnpir1N (double costh, void *para1, int *partid_d, int npt_max, int nphi_max, double *ptarray_d, double *phiarray_d, double* dN_ptdptdphidy_d)
{
    if(DEBUG <= 7) return 1.0;
    pblockN *para = (pblockN *) para1;
    double r;
    para->costh = costh;
    para->sinth = sqrt (1.0 - para->costh * para->costh);
    //Integrates the "dnpir2N" kernal over phi using gaussian integration
    r = gauss(PTN2, *dnpir2N, 0.0, 2.0 * PI, para, partid_d, npt_max, nphi_max, ptarray_d, phiarray_d, dN_ptdptdphidy_d);
    //if(isnan(r)) printf("dnpir1N r \n");
    return r;
}

__device__ double dn2ptN (double w2, void *para1, int *partid_d, int npt_max, int nphi_max, double *ptarray_d, double *phiarray_d, double *dN_ptdptdphidy_d)
{
    if(DEBUG <= 5) return 1.0;
    //double test;
    pblockN *para = (pblockN *) para1;
    para->e0 = (para->mr * para->mr + para->m1 * para->m1 - w2) / (2 * para->mr);
    para->p0 = sqrt (para->e0 * para->e0 - para->m1 * para->m1);
    //if(isnan(para->p0))
    //{
    //    printf("dn2ptN, %E %E %E \n", para->p0, para->e0, para->m1);
    //}
    //test = gauss(PTN1, *dnpir1N, -1.0, 1.0, para);
    //if(isnan(test))
    //{
    //    printf("gauss, %E \n", test);
    //}

    //Integrate the "dnpir1N" kernal over cos(theta) using gaussian integration
    //PTN1 is defined to be 8 in decay.cpp

    return gauss (PTN1, *dnpir1N, -1.0, 1.0, para, partid_d, npt_max, nphi_max, ptarray_d, phiarray_d, dN_ptdptdphidy_d);
}

__device__ double dn3ptN (double x, void* para1, int *partid_d, int npt_max, int nphi_max, double *ptarray_d, double *phiarray_d, double* dN_ptdptdphidy_d)
//The integration kernel for "W" in 3-body decays.
{
    if(DEBUG <= 6) return 1.0;
    pblockN *para = (pblockN *) para1;
    double e0 =(para->mr * para->mr + para->m1 * para->m1 - x) / (2 * para->mr);
    double p0 = sqrt (e0 * e0 - para->m1 * para->m1);
    double a = (para->m2 + para->m3) * (para->m2 + para->m3);
    double b = (para->m2 - para->m3) * (para->m2 - para->m3);
    double re = p0 * sqrt ((x - a) * (x - b)) / x * dn2ptN (x, para, partid_d, npt_max, nphi_max, ptarray_d, phiarray_d, dN_ptdptdphidy_d);
    return re;
}



__device__ double Edndp3_2bodyN (double y, double pt, double phi, double m1, double m2, double mr, int res_num, int *partid_d, int npt_max, int nphi_max, double *ptarray_d, double *phiarray_d, double* dN_ptdptdphidy_d)
// in units of GeV^-2,includes phasespace and volume, does not include degeneracy factors
// double y;  /* rapidity of particle 1 */
// double pt; /* transverse momentum of particle 1 */
// double phi; /* phi angle of particle 1 */
// double m1, m2; /* restmasses of decay particles in MeV */
// double mr;     /* restmass of resonance MeV            */
// int res_num;   /* Montecarlo number of the Resonance   */
{
    if(DEBUG <= 1) return 1.0;
    double mt = sqrt (pt * pt + m1 * m1);
    double norm2;                       /* 2-body normalization */
    pblockN para;
    double res2;

    para.pt  = pt;
    para.mt  = mt;
    para.e   = mt * cosh (y);
    para.pl  = mt * sinh (y);
    para.y   = y;
    para.phi = phi;
    para.m1  = m1;
    para.m2  = m2;
    para.mr  = mr;

    para.res_num = res_num;

    norm2 = 1.0 / (2.0 * PI);
    //Calls the integration routines for 2-body
    res2 = norm2 * dn2ptN (m2 * m2, &para, partid_d, npt_max, nphi_max, ptarray_d, phiarray_d, dN_ptdptdphidy_d);

    //if(isnan(res2))
    //{
    //    printf("res2, %E %E \n",res2, dn2ptN(m2*m2, &para));
    //}

    return res2;                        /* like Ed3ndp3_2body() */
}

__device__ double Edndp3_3bodyN (double y, double pt, double phi, double m1, double m2, double m3, double mr, double norm3, int res_num, int *partid_d, int npt_max, int nphi_max, double *ptarray_d, double *phiarray_d, double* dN_ptdptdphidy_d)
//in units of GeV^-2,includes phase space and volume, does not include degeneracy factors
{
    if(DEBUG <= 2) return 1.0;
    double mt = sqrt (pt * pt + m1 * m1);
    pblockN para;
    double wmin, wmax;
    double res3;
    double slope;                       /* slope of resonance for high mt */
    int pn;

    para.pt = pt;
    para.mt = mt;
    para.y = y;
    para.e = mt * cosh (y);
    para.pl = mt * sinh (y);
    para.phi = phi;

    para.m1 = m1;
    para.m2 = m2;
    para.m3 = m3;
    para.mr = mr;
    /*************Need partid_d as arg*/
    pn = partid_d[MHALF + res_num];

    para.res_num = res_num;

    wmin = (m2 + m3) * (m2 + m3);
    wmax = (mr - m1) * (mr - m1);
    //Integrates "W" using gaussian
    res3 = 2.0 * norm3 * gauss (PTS4, *dn3ptN, wmin, wmax, &para, partid_d, npt_max, nphi_max, ptarray_d, phiarray_d, dN_ptdptdphidy_d) / mr;
    return res3;
}


__global__ void add_reso_kernel( int pn, int pnR, int k, int j, int npt_max, int nphi_max, double* ptarray_d, double* phiarray_d, int numpart, double branch, int* decayPart_d, double *mass_d, double *width_d, int* partid_d, double* dN_ptdptdphidy_d, long int* monval_d)
{
	// int pn;                      /* internal number of daughter part. */
	// int pnR;                     /* internal number of Resonance */
	// int k;                       /* daughter-particle place in the decay array  */
	// int j;                       /* internal number of decay */
	// int numpart;			/* number of particles in the decay*/
	// int *decayPart_d;		/* pointer to array of montecarlo of decay particles*/
	// double* mass_d;		/* pointer to array of all masses*/
	// double* width_d		/* array pointing to widths of resonances*/
	// double* dN_ptdptdphidy_d	/* the particle spectrum */

	nblock paranorm;
	double y = 0.0;
    	double m1, m2, m3, mr;
    	double norm3;                   /* normalisation of 3-body integral */
    	int pn2, pn3, pn4;          	/* internal numbers for resonances */
    	//int part; //not used
    	//int l, i; //not used
    
	//These aren't necessary	
	//int npt, nphi;
	//npt = particle[pn].npt;
    	//nphi = particle[pn].nphi;

	// Determine the number of particles involved in the decay with the switch
    	switch (abs (/*particleDecay[j].*/numpart))
    	{
		case 1:
        	//Only 1 particle, if it gets here, by accident, this prevents any integration for 1 particle chains
            	break;

        	case 2: // 2-body decay
        	{
             		if(k == 0)
				pn2 = partid_d[ MHALF + decayPart_d[5*j+1] ];
                  		//pn2 = partid[MHALF + particleDecay[j].part[1]];
              		else
				pn2 = partid_d[ MHALF + decayPart_d[5*j] ];
                 		//pn2 = partid[MHALF + particleDecay[j].part[0]];

              		//printf ("case 2:  i %3i j %3i k %3i \n", pn, j, k);
              		m1 = mass_d[pn];
              		m2 = mass_d[pn2];
              		mr = mass_d[pnR];
              		while ((m1 + m2) > mr)
              		{
                  		mr += 0.25 * /*particle[pnR].*/ width_d[pnR];
                  		m1 -= 0.5 *  /*particle[pn].*/  width_d[pn];
                  		m2 -= 0.5 *  /*particle[pn2].*/ width_d[pn2];
              		}

			//thread dependent stuff
			//First let a block loop enough to cover all momemtum space
			int timesToLoop = (npt_max*nphi_max + threadsPerBlock - 1)/threadsPerBlock; 
			int localIdx = threadIdx.x;
			for(int loop = 0; loop < timesToLoop; loop++)
			{
				// Call the 2-body decay integral and add its contribution to the daughter particle of interest
				if(localIdx < npt_max*nphi_max){
				int ptIdx  = localIdx%npt_max;
				int phiIdx = localIdx/npt_max;
 			//	printf("2 body dN_.. += branch*Edn... %f, for ptIdx = %d, phiIdx = %d, pnR = %d\n", branch * Edndp3_2bodyN(y, ptarray_d[ptIdx], phiarray_d[phiIdx], m1, m2, mr, monval_d[pnR], partid_d, npt_max, nphi_max,ptarray_d, phiarray_d, dN_ptdptdphidy_d), ptIdx, phiIdx, pnR );

				dN_ptdptdphidy_d[pn*npt_max*nphi_max + phiIdx*npt_max + ptIdx] += branch * Edndp3_2bodyN(y, ptarray_d[ptIdx], phiarray_d[phiIdx], m1, m2, mr, monval_d[pnR], partid_d, npt_max, nphi_max,ptarray_d, phiarray_d, dN_ptdptdphidy_d );
				localIdx += blockDim.x;
			}}//have extra bracket for if(threadIdx.x ....)

			/*
              		for (l = 0; l < npt; l++)
              		{
                  		for (i = 0; i < nphi; i++)
                  		{
                		// Call the 2-body decay integral and add its contribution to the daughter particle of interest
                  		particle[pn].dNdptdphi[l][i] += particleDecay[j].branch *
                                                	Edndp3_2bodyN(y, particle[pn].pt[l], PHI[i],
                                                                    m1, m2, mr, particle[pnR].monval);
                  		}
              		}
			*/
              		break;
        	}
		case 3: //3-body decay
        	{
              		if (k == 0)
              		{
				pn2 = partid_d[ MHALF + decayPart_d[5*j+1] ];
                                pn3 = partid_d[ MHALF + decayPart_d[5*j+2] ];
                  		//pn2 = partid[MHALF + particleDecay[j].part[1]];
                  		//pn3 = partid[MHALF + particleDecay[j].part[2]];
              		}
              		else
              		{
                  		if (k == 1)
                  		{
					pn2 = partid_d[ MHALF + decayPart_d[5*j]   ];
                                	pn3 = partid_d[ MHALF + decayPart_d[5*j+2] ];
                      			//pn2 = partid[MHALF + particleDecay[j].part[0]];
                      			//pn3 = partid[MHALF + particleDecay[j].part[2]];
                  		}
                  		else
                  		{
					pn2 = partid_d[ MHALF + decayPart_d[5*j  ] ];
                                	pn3 = partid_d[ MHALF + decayPart_d[5*j+1] ];
                      			//pn2 = partid[MHALF + particleDecay[j].part[0]];
                      			//pn3 = partid[MHALF + particleDecay[j].part[1]];
                  		}
              		}

              		m1 = /*particle[pn].*/ mass_d[pn];
              		m2 = /*particle[pn2].*/mass_d[pn2];
              		m3 = /*particle[pn3].*/mass_d[pn3];
              		mr = /*particle[pnR].*/mass_d[pnR];
              		paranorm.a = (mr + m1) * (mr + m1);
              		paranorm.b = (mr - m1) * (mr - m1);
              		paranorm.c = (m2 + m3) * (m2 + m3);
              		paranorm.d = (m2 - m3) * (m2 - m3);
              		norm3 = mr * mr / (2 * PI * gauss (PTS3,   norm3int, paranorm.c,
                                           paranorm.b, &paranorm, partid_d, npt_max, nphi_max, ptarray_d, phiarray_d, dN_ptdptdphidy_d));

              		//printf("case 3:  i %3i j %3i k %3i \n",pn,j,k);
	 	        int timesToLoop = (npt_max*nphi_max + threadsPerBlock - 1)/threadsPerBlock;
                        int localIdx = threadIdx.x;
                        for(int loop = 0; loop < timesToLoop; loop++)
                        {
                                // Call the 2-body decay integral and add its contribution to the daughter particle of interest
                                if(localIdx < npt_max*nphi_max){
                                int ptIdx  = localIdx%npt_max;
                                int phiIdx = localIdx/npt_max;
			//	printf("3 body dN_.. += branch*Edn... %f, for ptIdx = %d, phiIdx = %d, pnR = %d\n", branch * Edndp3_3bodyN(y, ptarray_d[ptIdx], phiarray_d[phiIdx], m1, m2, m3, mr, norm3, monval_d[pnR], partid_d, npt_max, nphi_max, ptarray_d, phiarray_d, dN_ptdptdphidy_d), ptIdx, phiIdx, pnR );

                                dN_ptdptdphidy_d[pn*npt_max*nphi_max + phiIdx*npt_max + ptIdx] += branch * Edndp3_3bodyN(y, ptarray_d[ptIdx], phiarray_d[phiIdx], m1, m2, m3, mr, norm3, monval_d[pnR], partid_d, npt_max, nphi_max, ptarray_d, phiarray_d, dN_ptdptdphidy_d);

				localIdx += blockDim.x;
                        }}//have extra bracket for if(threadIdx.x ....)

			/*for (i = 0; i < nphi; i++)
              		{
                  		for (l = 0; l < npt; l++)
                  		{
                    		//Call the 3-body decay integral and add its contribution to the daughter particle of interest
                    			particle[pn].dNdptdphi[l][i] += particleDecay[j].branch *
                                                  Edndp3_3bodyN (y, particle[pn].pt[l], PHI[i],
                                                      m1, m2, m3, mr, norm3, particle[pnR].monval);
                  		}
              		}*/
              		break;
        	}

	        case 4: //4-body decay (rare and low contribution)
        	{
              		if (k == 0)
              		{
				pn2 = partid_d[ MHALF + decayPart_d[5*j + 1] ];
                                pn3 = partid_d[ MHALF + decayPart_d[5*j + 2] ];
                                pn4 = partid_d[ MHALF + decayPart_d[5*j + 3] ];
                  		//pn2 = partid[MHALF + particleDecay[j].part[1]];
                  		//pn3 = partid[MHALF + particleDecay[j].part[2]];
                  		//pn4 = partid[MHALF + particleDecay[j].part[3]];
              		}
              		else
              		{
                  		if (k == 1)
                  		{
					pn2 = partid_d[ MHALF + decayPart_d[5*j + 0] ];
                               	 	pn3 = partid_d[ MHALF + decayPart_d[5*j + 2] ];
                                	pn4 = partid_d[ MHALF + decayPart_d[5*j + 3] ];
                  			//pn2 = partid[MHALF + particleDecay[j].part[0]];
                  			//pn3 = partid[MHALF + particleDecay[j].part[2]];
                  			//pn4 = partid[MHALF + particleDecay[j].part[3]];
                  		}
                  		else
                  		{
                  			if (k == 2)
                  			{
						pn2 = partid_d[ MHALF + decayPart_d[5*j + 0] ];
                               	 		pn3 = partid_d[ MHALF + decayPart_d[5*j + 1] ];
                                		pn4 = partid_d[ MHALF + decayPart_d[5*j + 3] ];
                      				//pn2 = partid[MHALF + particleDecay[j].part[0]];
                     				//pn3 = partid[MHALF + particleDecay[j].part[1]];
                     	 			//pn4 = partid[MHALF + particleDecay[j].part[3]];
                  			}
                  			else
                  			{
						pn2 = partid_d[ MHALF + decayPart_d[5*j + 0] ];
                               	 		pn3 = partid_d[ MHALF + decayPart_d[5*j + 1] ];
                                		pn4 = partid_d[ MHALF + decayPart_d[5*j + 2] ];
                      				//pn2 = partid[MHALF + particleDecay[j].part[0]];
                      				//pn3 = partid[MHALF + particleDecay[j].part[1]];
                      				//pn4 = partid[MHALF + particleDecay[j].part[2]];
                  			}
                  		}
              		}
              		//approximate the 4-body with a 3-body decay with the 4th particle being the center of mass of 2 particles.
              		m1 = /*particle[pn].*/ mass_d[pn];
              		m2 = /*particle[pn2].*/mass_d[pn2];
              		mr = /*particle[pnR].*/mass_d[pnR];
              		m3 = 0.5 * (/*particle[pn3].*/mass_d[pn3] + /*particle[pn4].*/mass_d[pn4] + mr - m1 - m2);
              		paranorm.a = (mr + m1) * (mr + m1);
              		paranorm.b = (mr - m1) * (mr - m1);
              		paranorm.c = (m2 + m3) * (m2 + m3);
              		paranorm.d = (m2 - m3) * (m2 - m3);
              		norm3 = mr * mr / (2 * PI * gauss (PTS3, norm3int, paranorm.c,
                                           paranorm.b, &paranorm, partid_d, npt_max, nphi_max, ptarray_d, phiarray_d, dN_ptdptdphidy_d));
              		//printf("case 3:  i %3i j %3i k %3i \n",pn,j,k);

			int timesToLoop = (npt_max*nphi_max + threadsPerBlock - 1)/threadsPerBlock;
                        int localIdx = threadIdx.x;
                        for(int loop = 0; loop < timesToLoop; loop++)
                        {
				//the 4-body particleDecay approximated by the 3-body decay routine
                                if(localIdx < npt_max*nphi_max){
                                int ptIdx  = localIdx%npt_max;
                                int phiIdx = localIdx/npt_max;
                                dN_ptdptdphidy_d[pn*npt_max*nphi_max + phiIdx*npt_max + ptIdx] += branch * Edndp3_3bodyN(y, ptarray_d[ptIdx], phiarray_d[phiIdx], m1, m2, m3, mr, norm3, monval_d[pnR], partid_d, npt_max, nphi_max, ptarray_d, phiarray_d, dN_ptdptdphidy_d);
                                localIdx += blockDim.x;
                        }}

              		/*for (i = 0; i < nphi; i++)
              		{
                  		for (l = 0; l < npt; l++)
                  		{
                    			//the 4-body particleDecay approximated by the 3-body decay routine
                  			particle[pn].dNdptdphi[l][i] += particleDecay[j].branch *
                                                  )Edndp3_3bodyN(y, particle[pn].pt[l], PHI[i],
                                                                    m1, m2, m3, mr, norm3, particle[pnR].monval);
                  		}
              		}*/
              		break;
        	}
        	default:
            		printf ("ERROR in add_reso! \n");
           		printf ("%i decay not implemented ! \n", abs (numpart));
            		//exit (0); //Cant call from device

	}//end switch
	return;
}

void cal_reso_decays_GPU (int maxpart, int maxdecay, int bound)
{
	// int maxpart; the particle number of the maximum particle resonances can decay to.
	// int maxdecay; the particle number of the maximum thing that can decay
	// int bound; 	particle number value of a lower bound in the spectrum of particles taken into account
	// ^none are montecarlo, all array indices
	
	//Need a lot of arrays to pass to GPU
	//First, all properties fromparticleDecay[]
	//int    decayReso[NUMDECAY];
	//int    decayNumpart[NUMDECAY];
	//double decayBranch[NUMDECAY];
	int    decayPart[5*NUMDECAY];
	
	//Properties from particle[]
	long int particleMonval[NUMPARTICLE];
	//char     particleName[26*NUMPARTICLE];
	double   particleMass[NUMPARTICLE];
	double   particleWidth[NUMPARTICLE];
	//int	 particleGspin[NUMPARTICLE];
	//int      particleBaryon[NUMPARTICLE];
	//int      particleStrange[NUMPARTICLE];
	//int      particleCharm[NUMPARTICLE];
	//int      particleBottom[NUMPARTICLE];
	//int      particleGisospin[NUMPARTICLE];
	//int      particleCharge[NUMPARTICLE];
	//int      particleDecays[NUMPARTICLE];
	//int      particleStable[NUMPARTICLE];
	
	//Need a spectrum for all particles
	double *dN_ptdptdphidy;
	//Use a particle that will have a spectrum
	int npt_max  = particle[11].npt;
	int nphi_max = particle[11].nphi;
	double ptarray[npt_max];
	double phiarray[nphi_max];
	dN_ptdptdphidy = (double*) malloc( NUMPARTICLE*npt_max*nphi_max*sizeof(double) );
	
	//Fill arrays
	for(int i = 0; i< NUMDECAY; i++)
	{
		for(int j=0; j<5; j++)
		decayPart[5*i+j] = particleDecay[i].part[j];

	}

	for(int i=0; i< npt_max; i++) ptarray[i] = particle[21].pt[i];
	for(int i=0; i< nphi_max;i++) phiarray[i] = PHI[i];

	for(int i=0; i< NUMPARTICLE; i++)
	{
		particleMonval[i] = particle[i].monval;
		particleWidth[i]  = particle[i].width;
		particleMass[i]   = particle[i].mass;

		for(int j=0; j< nphi_max; j++)
		for(int k=0; k< npt_max;  k++)
		dN_ptdptdphidy[i*npt_max*nphi_max + j*npt_max + k] = particle[i].dNdptdphi[k][j];
	}

	//device copies of useful stuff
	int	*decayPart_d;
	long int *monval_d;
	double  *particleWidth_d;
	double 	*particleMass_d;
	int 	*partid_d;
	double 	*dN_ptdptdphidy_d;
	double  *ptarray_d;
	double  *phiarray_d;

	//Allocate memory
	cudaMalloc( (void**) &monval_d, 		NUMPARTICLE*sizeof(long int) );
	cudaMalloc( (void**) &decayPart_d, 		5*NUMDECAY*sizeof(int) );
	cudaMalloc( (void**) &particleWidth_d,		NUMPARTICLE*sizeof(double) );
	cudaMalloc( (void**) &particleMass_d, 		NUMPARTICLE*sizeof(double) );
	cudaMalloc( (void**) &partid_d, 		MAXINTV*sizeof(int) );
	cudaMalloc( (void**) &dN_ptdptdphidy_d, 	NUMPARTICLE*npt_max*nphi_max*sizeof(double) );
	cudaMalloc( (void**) &ptarray_d,		npt_max*sizeof(double) );
	cudaMalloc( (void**) &phiarray_d,		nphi_max*sizeof(double) );

	//Copy memory
	cudaMemcpy( monval_d,		particleMonval, NUMPARTICLE*sizeof(long int),   cudaMemcpyHostToDevice );
	cudaMemcpy( decayPart_d, 	decayPart, 	5*NUMDECAY*sizeof(int), 	cudaMemcpyHostToDevice );
	cudaMemcpy( particleWidth_d, 	particleWidth, 	NUMPARTICLE*sizeof(double), 	cudaMemcpyHostToDevice );
	cudaMemcpy( particleMass_d, 	particleMass, 	NUMPARTICLE*sizeof(double), 	cudaMemcpyHostToDevice );
	cudaMemcpy( partid_d,		partid,		MAXINTV*sizeof(int),		cudaMemcpyHostToDevice );
	cudaMemcpy( dN_ptdptdphidy_d, 	dN_ptdptdphidy, NUMPARTICLE*npt_max*nphi_max*sizeof(double), cudaMemcpyHostToDevice );
	cudaMemcpy( ptarray_d, 		ptarray, 	npt_max*sizeof(double), 		cudaMemcpyHostToDevice );
	cudaMemcpy( phiarray_d,		phiarray,	nphi_max*sizeof(double),		cudaMemcpyHostToDevice );

    int i, j, k; // l, ll not used, so deleted
    int pn, pnR, pnaR;
    int part;

//for(int ab = 0; ab<NUMPARTICLE; ab++)
//{
//printf("%f, %s\n", dN_ptdptdphidy[ab*npt_max*nphi_max], particle[ab].name);
//}

    printf (" CALCULATE RESONANCE DECAYS (on GPU) \n");
    pn = partid[MHALF + bound];
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for(i=maxpart-1;i > pn-1;i--)  //Cycle the particles known from the resoweak.dat input
    {
	part = particle[i].monval;
        printf ("Calculating the decays with ");
        printf ("%s \n", particle[i].name);
        printf ("%i ", part);
        // Check to see whether or not the particle is baryon, anti-baryon or meson
	//*****************************
	cudaDeviceSynchronize(); //Make sure all decays to this particle are done
	//*****************************
        switch (particle[i].baryon)
	{
		case 1: //Baryon
              	{
                  	printf("is a baryon. \n");
                	// Cycle through every decay channel known (as given in pdg.dat)
                	// to see if the particle was a daughter particle in a decay channel
			
                	for (j = 0; j < maxdecay; j++)
                  	{
                   		pnR = partid[MHALF + particleDecay[j].reso];
                        	//printf("Partid is %i.\n",pnR);
                        	for(k = 0; k < abs (particleDecay[j].numpart); k++)
                        	{
                        		// Make sure that the particle decay channel isn't trivial and contains the daughter particle
                            		if((part == particleDecay[j].part[k]) && (particleDecay[j].numpart != 1))
                              		// printf("Calculating a decay \n");
                              		add_reso_kernel<<<1,threadsPerBlock>>>(i, pnR, k, j, npt_max, nphi_max, ptarray_d, phiarray_d, particleDecay[j].numpart, particleDecay[j].branch, decayPart_d, particleMass_d, particleWidth_d, partid_d, dN_ptdptdphidy_d, monval_d);
                        	}
                    	}
                  break;
              	}

	 	case -1: //Anti-Baryon
              	{
                	printf("is an anti-baryon.\n");
                	// Cycle through every decay channel known (as given in pdg.dat)
                	// to see if the particle was a daughter particle in a decay channel
                  	for(j = 0; j < maxdecay; j++)
                  	{
                        	pnaR = partid[MHALF - particleDecay[j].reso];
                        	//printf("Partid is %i.\n",pnaR);
                        	for (k = 0; k < abs (particleDecay[j].numpart); k++)
                        	{
                        		//Make sure that the decay channel isn't trivial and contains the daughter particle
                            		if((-part == particleDecay[j].part[k]) && (particleDecay[j].numpart != 1))
                            		{
                                		//printf("Calculating a decay \n");
                                		add_reso_kernel<<<1,threadsPerBlock>>>(i, pnaR, k, j, npt_max, nphi_max, ptarray_d, phiarray_d, particleDecay[j].numpart, particleDecay[j].branch, decayPart_d, particleMass_d, particleWidth_d, partid_d, dN_ptdptdphidy_d, monval_d);
                           		}
                        	}
                  	}
                  break;
              	}
	      

	        case 0:// Meson
              	{
                  	printf("is a meson. \n");

                  	for (j = 0; j < maxdecay; j++)
                  	{
                        	pnR = partid[MHALF + particleDecay[j].reso];
                        	//printf("Partid is %i.\n",pnR);
                        	for(k = 0; k < abs (particleDecay[j].numpart); k++)
                        	{
                            		if(particle[pnR].baryon == 1)
                            		{
                                  		pnaR = partid[MHALF - particleDecay[j].reso];
                                  		if((particle[i].charge == 0) && (particle[i].strange == 0))
                                  		{
                                       			if((part == particleDecay[j].part[k]) && (particleDecay[j].numpart != 1))
                                       			{
                                             			//printf("Calculating a decay \n");
                                             			add_reso_kernel<<<1,threadsPerBlock>>>(i, pnR, k, j, npt_max, nphi_max, ptarray_d, phiarray_d, particleDecay[j].numpart, particleDecay[j].branch, decayPart_d, particleMass_d, particleWidth_d, partid_d, dN_ptdptdphidy_d, monval_d);
                                             			add_reso_kernel<<<1,threadsPerBlock>>>(i, pnaR, k, j, npt_max, nphi_max, ptarray_d, phiarray_d, particleDecay[j].numpart, particleDecay[j].branch, decayPart_d, particleMass_d, particleWidth_d, partid_d, dN_ptdptdphidy_d, monval_d);
                                           		}
                                  		}
                                  		else
                                  		{
                                     			if ((part == particleDecay[j].part[k]) && (particleDecay[j].numpart != 1))
                                      			{
                                             			//printf("Calculating a decay \n");
                                     				add_reso_kernel<<<1,threadsPerBlock>>>(i, pnR, k, j, npt_max, nphi_max, ptarray_d, phiarray_d, particleDecay[j].numpart, particleDecay[j].branch, decayPart_d, particleMass_d, particleWidth_d, partid_d, dN_ptdptdphidy_d, monval_d);
                                			}
                                			if ((-part == particleDecay[j].part[k]) && (particleDecay[j].numpart != 1))
                                      			{
                                            			//printf("Calculating a decay \n");
                                    				add_reso_kernel<<<1,threadsPerBlock>>>(i, pnaR, k, j, npt_max, nphi_max, ptarray_d, phiarray_d, particleDecay[j].numpart, particleDecay[j].branch, decayPart_d, particleMass_d, particleWidth_d, partid_d, dN_ptdptdphidy_d, monval_d);
                                			}
                                  		}
                            		}
					else
                            		{
                                  		if((part == particleDecay[j].part[k]) && (particleDecay[j].numpart != 1))
                                  		{
                                      			//printf("Calculating a decay \n");
                                      			add_reso_kernel<<<1,threadsPerBlock>>>(i, pnR, k, j, npt_max, nphi_max, ptarray_d, phiarray_d, particleDecay[j].numpart, particleDecay[j].branch, decayPart_d, particleMass_d, particleWidth_d, partid_d, dN_ptdptdphidy_d, monval_d);
                                  		}
                            		}
                        	}
                  	}
                  break;
              }
              default:
                printf ("Error in switch in func partden_wdecay \n");
                //exit (0); //Cant call from device


	}//switch statement
    }//for loop


    cudaDeviceSynchronize(); //Make sure all decays to this particle are done
    //Put the spectrum into the format for writing.
    cudaMemcpy( dN_ptdptdphidy,   dN_ptdptdphidy_d, NUMPARTICLE*npt_max*nphi_max*sizeof(double), cudaMemcpyDeviceToHost );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("This took %f milliseconds.\n", milliseconds);

//for(int ab = 0; ab<NUMPARTICLE; ab++)
//{
//printf("%f, %s\n", dN_ptdptdphidy[ab*npt_max*nphi_max], particle[ab].name);
//}

    for(int a = 0; a<NUMPARTICLE; a++ )
    for(int b = 0; b<nphi_max;     b++ )
    for(int c = 0; c<npt_max;	  c++ )
	particle[a].dNdptdphi[c][b] = dN_ptdptdphidy[a*npt_max*nphi_max + b*npt_max + c];


    //Free memory
    free( dN_ptdptdphidy );
    cudaFree( monval_d );
    cudaFree( decayPart_d );
    cudaFree( particleWidth_d );
    cudaFree( particleMass_d );
    cudaFree( partid_d );
    cudaFree( dN_ptdptdphidy_d );
    cudaFree( ptarray_d );
    cudaFree( phiarray_d );
}

#endif
