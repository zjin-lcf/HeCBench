void 
kernel_wrapper(	fp* image,											// input image
		int Nr,												// IMAGE nbr of rows
		int Nc,												// IMAGE nbr of cols
		long Ne,											// IMAGE nbr of elem
		int niter,											// nbr of iterations
		fp lambda,											// update step size
		long NeROI,											// ROI nbr of elements
		int* iN,
		int* iS,
		int* jE,
		int* jW,
		int iter);											

