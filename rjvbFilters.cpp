#include "rjvbFilters.c"

void matrix_inverse(double *Min, double *Mout, int actualsize)
{
    // matrix_inverse(double *Min, double *Mout, int actualsize)
	
    /* Loop variables */
    int i, j, k;
    /* Sum variables */
    double sum,x;
    
    /*  Copy the input matrix to output matrix */
    for(i=0; i<actualsize*actualsize; i++) { Mout[i]=Min[i]; }
    
    /* Add small value to diagonal if diagonal is zero */
    for(i=0; i<actualsize; i++)
    { 
        j=i*actualsize+i;
        if((Mout[j]<1e-12)&&(Mout[j]>-1e-12)){ Mout[j]=1e-12; }
    }
    
    /* Matrix size must be larger than one */
    if (actualsize <= 1) return;
    
    for (i=1; i < actualsize; i++) {
        Mout[i] /= Mout[0]; /* normalize row 0 */
    }
    
    for (i=1; i < actualsize; i++)  {
        for (j=i; j < actualsize; j++)  { /* do a column of L */
            sum = 0.0;
            for (k = 0; k < i; k++) {
                sum += Mout[j*actualsize+k] * Mout[k*actualsize+i];
            }
            Mout[j*actualsize+i] -= sum;
        }
        if (i == actualsize-1) continue;
        for (j=i+1; j < actualsize; j++)  {  /* do a row of U */
            sum = 0.0;
            for (k = 0; k < i; k++) {
                sum += Mout[i*actualsize+k]*Mout[k*actualsize+j];
            }
            Mout[i*actualsize+j] = (Mout[i*actualsize+j]-sum) / Mout[i*actualsize+i];
        }
    }
    for ( i = 0; i < actualsize; i++ )  /* invert L */ {
        for ( j = i; j < actualsize; j++ )  {
            x = 1.0;
            if ( i != j ) {
                x = 0.0;
                for ( k = i; k < j; k++ ) {
                    x -= Mout[j*actualsize+k]*Mout[k*actualsize+i];
                }
            }
            Mout[j*actualsize+i] = x / Mout[j*actualsize+j];
        }
    }
    for ( i = 0; i < actualsize; i++ ) /* invert U */ {
        for ( j = i; j < actualsize; j++ )  {
            if ( i == j ) continue;
            sum = 0.0;
            for ( k = i; k < j; k++ ) {
                sum += Mout[k*actualsize+j]*( (i==k) ? 1.0 : Mout[i*actualsize+k] );
            }
            Mout[i*actualsize+j] = -sum;
        }
    }
    for ( i = 0; i < actualsize; i++ ) /* final inversion */ {
        for ( j = 0; j < actualsize; j++ )  {
            sum = 0.0;
            for ( k = ((i>j)?i:j); k < actualsize; k++ ) {
                sum += ((j==k)?1.0:Mout[j*actualsize+k])*Mout[k*actualsize+i];
            }
            Mout[j*actualsize+i] = sum;
        }
    }
}

int altSavGol2D(double *h, const int col, const int row, const int px, const int py)
//void SavGol(double *h, int _col, int _row, int _px, int _py)
{
	/* This function generates Savitzky-Golay 2D filter
	 *
	 * SavGol(double *h, int _col, int _row, int _px, int _py);
	 * h   : Pointer to Output (empty) memory space with size of _col*_row [1D array]
	 * _col: column size of SG filter
	 * _row: row size of SG filter (same as _col)
	 * _px : order number of column (less value than _col)
	 * _py : order number of row	(less value than _row) 
     *
	 */
//	if (_col==3 && _col<=_px){
	if (col==3 && col<=px) {
		return 0;
	}
	
 //   const int col = _col, row = _row, px = _px, py = _py;	
	double *X = new double [col*row];
	double *Y = new double [col*row];
	double *A = new double [col*row*(px*2+1)];	// Matrix A
	double *A_ = new double [col*row*(px*2+1)];// Transpose of Matrix A 
	double **tempA, **tempA_;
	double **matMul, **matMul_;
	double **invMat2D;
	double *matMul1D = new double [(px*2+1)*(px*2+1)]; // matmul to 1D array 
	double *invMat = new double [(px*2+1)*(px*2+1)];   //1D array inverse matrix
	int i, j;
	
	// calloc «‘ºˆ: µø¿˚ «“¥Á π◊ '0'¿∏∑Œ πËø≠ √ ±‚»≠
	tempA = (double **) calloc(col*row, sizeof(double));
	for (i=0; i<col*row; i++)	tempA[i] = (double *) calloc(px*2+1, sizeof(double));
	// 'new'∏¶ ¿ÃøÎ«ÿ µø¿˚ «“¥Á
	tempA_ = new double *[px*2+1];
	for (i=0; i<px*2+1; i++)  tempA_[i] = new double [col*row];
	// for loop∑Œ √ ±‚»≠
	for (i=0; i<px*2+1; i++)  for (j=0; j<col*row; j++) tempA_[i][j] = 0.0;
	
	matMul = (double **) calloc((px*2+1), sizeof(double));
	for (i=0; i<px*2+1; i++) matMul[i] = (double *) calloc((px*2+1), sizeof(double));
	
	matMul_ = new double *[px*2+1];
	for (i=0; i<px*2+1; i++) matMul_[i] = new double [col*row];
	for (i=0; i<px*2+1; i++) for (j=0; j<col*row; j++) matMul_[i][j] = 0.0;
	
	invMat2D = new double *[px*2+1];
	for (i=0; i<px*2+1; i++) invMat2D[i] = new double [px*2+1];
	for (i=0; i<px*2+1; i++) for (j=0; j<px*2+1; j++) invMat2D[i][j] = 0.0;
	
	memset(A, 0, sizeof(double)*col*row*(px*2+1));
	memset(A_, 0, sizeof(double)*(px*2+1));
	memset(X, 0, sizeof(double)*(col*row));
	memset(Y, 0, sizeof(double)*(col*row));
	memset(invMat, 0, sizeof(double)*((px*2+1)*(px*2+1)));
	memset(h, 0, sizeof(double)*(col*row));
	
	int k_1=0;
	for (int i=0; i<col; i++){
		for (int j=0; j<col; j++){		
			X[k_1] = -(col-1)/2 + i;
			k_1++;
		}}
	
	int k_2 = 0;
	for (int j=0; j<row; j++){
		for (int i=0; i<row; i++){
			Y[k_2] = -(col-1)/2 + i;
			k_2++;
		}}
	
	int k_3=0;
	for (int k1=0; k1<2*px+1; k1++){
		for (int k2=0; k2<col*row ; k2++){
			if ( k1 < px )	A[k_3]= pow(X[k2],(px-k1));
			else if ( k1 >= px ) A[k_3]= pow(Y[k2],(py-k1+py));
			k_3++;
		}}
	
	int k_4 = 0;
	for (int j=0; j<px*2+1; j++){
		for (int i=0; i<col*row; i++){
			tempA[i][j] = A[k_4];		// converting 1D to 2D  		
			tempA_[j][i] = tempA[i][j]; // transpose of matrix A
			k_4++;
		}}
	
	for (int i=0; i<px*2+1; i++){
		for (int j=0; j<px*2+1; j++){
			for (int k=0; k<col*row; k++) matMul[i][j] += tempA_[i][k]*tempA[k][j];		// multiplication of above matrices
		}}
	
	int k_5 = 0;
	for (int j=0; j<px*2+1; j++){
		for (int i=0; i<px*2+1; i++){
			matMul1D[k_5]=matMul[i][j];			// converting 2D matmul to 1D array
			k_5++;
		}}
	
	matrix_inverse(matMul1D,invMat, px*2+1);
	
	int k = 0;
	for (int j=0; j<px*2+1; j++){
		for (int i=0; i<px*2+1; i++){
			invMat2D[i][j] = invMat[k];			// 1D array to 2D of invMat
			k++;
		}}
	
	for (int i=0; i<px*2+1; i++){
		for (int j=0; j<col*row; j++){
			for (int k=0; k<px*2+1; k++) matMul_[i][j] += invMat2D[i][k]*tempA_[k][j];		// multiplication of invMat2D and tempA_	
		}}
	
	for(int i=0; i<col*row; i++){
		h[i]=matMul_[(px*2+1)-1][i];
		//	printf("%lf \n", h[i]);										// h = Savitzky-Golay Filters
	}
	
	////////////////////////////deallocation memory//////////////////////////////////////////////
	if (X) delete[] X;
	if (Y) delete[] Y;
	if (A) delete[] A;
	if (A_) delete[] A_;
	
	if (tempA)
	{
		for (int i=0; i<col*row; i++) free(tempA[i]);	
		free(tempA);
	}
	
	if (tempA_)
	{
		for (int i=0; i<px*2+1; i++)
		{
			delete[] tempA_[i];
			tempA_[i] = NULL;
		}
		delete tempA_;
		tempA_ = NULL;
	}
	
	for (i=0; i<px*2+1; i++) free(matMul[i]);
	free(matMul);
	
	if (matMul_)
	{
		for(int i=0; i<px*2+1; i++)
		{
			delete[] matMul_[i];
			matMul_[i]= NULL;
		}
		delete matMul_;
		matMul_= NULL;
	}
	
	if (invMat2D)
	{
		for (int i=0; i<px*2+1; i++)
		{
			delete[] invMat2D[i];
			invMat2D[i] = NULL;
		}
		delete invMat2D;
		invMat2D = NULL;
	}
	
	if (invMat) delete[] invMat;
	if (matMul1D) delete[] matMul1D;
	return 1;
}

