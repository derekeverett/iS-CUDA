/* tools.c            functions for reso.c main program calculations 
*   Josef Sollfrank                  Nov. .98             */


#include	<string.h>
#include	<stdio.h>
#include	<cmath>

#include	"tools.h"

/* converts an integer to a string !*/

__device__ double lin_int(double x1, double x2, double f1, double f2, double x)
{
     if(DEBUG <= 10) return 1.0;

     double aa, bb, cc;
     double eps=1e-100;

       //Don't use these. They lead to bad approximations at high pt.
       //if (x < x1) return f1;   //avoid NAN CSHEN
       //if (x > x2) return f2;
	 if (x2 == x1) 
           aa = 0.0;
	 else
           aa =(f2-f1)/(x2-x1);
       bb = f1 - aa * x1;
       cc = aa*x + bb;

       //used for debugging
       //printf("lin_int(....) = %f, x1 = %f, x2 = %f, f1 = %f, f2 = %f, x =%f\n", cc,x1,x2,f1,f2,x);         
       return cc;
}


__device__ int poweri(int xx, int l)
	{ 
	 int i;
	 double re ;

	 re = xx ;
	 if (l == 0) return 1;
	 else {
	   for(i=1;i<l;i++) re = re*xx ; 
	   return re ;
	   }
	 }


__device__ void  convei(int zahl, char p[SSL])
	{
	  int dig, j, ii = 0, sta, ani, z ;
          double dum;

	  if (zahl < 0) {p[0] = 45; ++ii; z = -1.0*zahl;} //fabs not allowed on device 
	  //z = fabs(zahl);

	  dum = log10( (double) z);
          ani = (int) dum;
	  sta = poweri(10,ani);

	  for(j=ii; j <= ani+ii; j++) {
	     dig = z / sta ;
	     p[j]  = dig + 48 ;
	     z = z - dig*sta;
	     sta = sta / 10 ;
	  }
	  p[j] = '\0';
	}


