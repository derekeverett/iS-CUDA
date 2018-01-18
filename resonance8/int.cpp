/*	int.c
	routines for numerical integration
	started 1 June 90, es using old FORTRAN stuff and newer C stuff
*/

// This file contains all the integration routines needed.

#include	<stdio.h>
#include	<math.h>
#include        <stdlib.h>
#include	"int.h"

#ifdef TEST

double	testfunc();
double	testgalafunc();

main() {
	testgala();
	}

int	matherr() {		/* for debugging */
	printf("matherr\n");	
	}
	
testgauss() {
	double	xlo = 0, xhi = 0.99;
	printf( "quadrature gives exact %10f:\n", asin(xhi) - asin(xlo) );
	printf( "%2dpoints: %10f\n", 4,gauss( 4,testfunc,xlo,xhi) );
	printf( "%2dpoints: %10f\n", 8,gauss( 8,testfunc,xlo,xhi) );
	printf( "%2dpoints: %10f\n",10,gauss(10,testfunc,xlo,xhi) );
	printf( "%2dpoints: %10f\n",12,gauss(12,testfunc,xlo,xhi) );
	printf( "%2dpoints: %10f\n",16,gauss(16,testfunc,xlo,xhi) );
	printf( "%2dpoints: %10f\n",20,gauss(20,testfunc,xlo,xhi) );
	printf( "%2dpoints: %10f\n",48,gauss(48,testfunc,xlo,xhi) );
	}

testgala() {
	printf( "testing Gauss-Laguerre integration with sin(2x)*exp(-2x)\n" );
	printf( "quadrature gives exact %10f:\n", 0.25 );
	printf( "%2dpoints: %10f\n", 4, gala( 4,testgalafunc,0.0,0.5) ); 
	printf( "%2dpoints: %10f\n", 8, gala( 8,testgalafunc,0.0,0.5) );
	printf( "%2dpoints: %10f\n",12, gala(12,testgalafunc,0.0,0.5) );
	}

testgahe() {
	printf( "testing Gauss-Hermite integration with exp(-x*x)\n" );
	printf( "exact result: %10f\n", sqrt( M_PI ) );
	printf( "%2d points: %10f\n", 4, gahe( 4,testfunc,0.0,1.0) );
	printf( "%2d points: %10f\n", 8, gahe( 8,testfunc,0.0,1.0) );
	printf( "%2d points: %10f\n",16, gahe(16,testfunc,0.0,1.0) );
	}

testgauche() {
	double	xpole = 100, xbase = 0;
	printf("%.15f\n", asin(1.0) * 2 );
	printf( "testing Gauss-Chebyshev integration with 1/sqrt(1-x*x)\n" );
	printf( "exact result:%10f\n", asin(xpole)-asin(xbase) );
	printf( "  %2d points: %10f\n", 4, gauche( 4,testfunc,xbase,xpole) );
	printf( "  %2d points: %10f\n", 8, gauche( 8,testfunc,xbase,xpole) );
	printf( "  %2d points: %10f\n",16, gauche(16,testfunc,xbase,xpole) );
	printf( "  %2d points: %10f\n",96, gauche(96,testfunc,xbase,xpole) );
	}

double	testgalafunc( x )
	double	x;
	{
	return ( sin(2*x)*exp(-2*x) );
	}

double testfunc( x )
	double x;
	{
	/* x += 0.5; return exp( -x*x );	*/
	return 1/sqrt(100*100-x*x);
	}
#endif

#include "CUDA_int.cu"
