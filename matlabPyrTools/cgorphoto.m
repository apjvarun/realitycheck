%%% How Realistic is Photorealistic?
%%% S. Lyu and H. Farid 
%%% IEEE Transactions on Signal Processing, 2005 
%%% http://www.cs.dartmouth.edu/farid/publications/sp05b.html


%%% In addition to this code, you need the matlabPyrTools by
%%% E.P. Simoncelli:
%%%    http://www.cns.nyu.edu/~lcv/software.html
%%% 
%%% and a few functions from the MatLab image processing and 
%%% statstics toolboxes.
%%%
%%% To classify images as CG or Photo, based on these statistics,
%%% you need the support vector machine (SVM) code from:
%%%    http://www.csie.ntu.edu.tw/~cjlin/libsvm/
%%%
%%% or the linear discriminant analysis (LDA) code from:
%%%    http://www.cs.dartmouth.edu/farid/research/steg.m


%%% Posted on 1.14.2005


%%% Copyright (c), 2005, Trustees of Dartmouth College. All rights reserved.


%%% ----------------------------------------------------------------
%%% Takes as input a grayscale or color image, and returns a 72-D or
%%% 216-D statistical feature vector, respectively
%%%
function [ftr] = cgorphoto( im )

   im  = double( im );
   sz  = size(im);
   if min(sz(1,2))<256
      error( 'Image must be in size at least 256 by 256' );
      return;
   end
   if( ndims(im) == 3 )
      TYPE = 'c'; % color image
   else
      TYPE = 'g'; % grayscale image
   end
   
   im  = im_center_crop(im,256,256);
   ftr = wvltftr(im,3,TYPE);
   return;
   
%%% ----------------------------------------------------------------
%%% Collect either grayscale or color statistics
%%%
function [ftr] = wvltftr(im,lev,type)

   if type == 'g' % graylevel image features
      W = wvlt_decompose(im,lev+1); % wavelet decomposition
      ftr = stat_ftr(W); % statistical feature vector
   elseif type == 'c' % color image features
      W(1) = wvlt_decompose(im(:,:,1),lev+1); % wavelet decomposition
      W(2) = wvlt_decompose(im(:,:,2),lev+1);
      W(3) = wvlt_decompose(im(:,:,3),lev+1);
      ftr = stat_ftr(W); % statistical feature vector
   end
   return;
   
%%% ----------------------------------------------------------------
%%% Wavelet decomposition
%%%
function [W] = wvlt_decompose(im,lev)

   im        = im - min(im(:));
   im        = 255 / max(im(:)) * im; % normalize into [0,255]
   [pyr,ind] = buildWpyr(im,lev); % build wavelet pyramid
   
   for k = 1:lev
      [lev,sz]  = wpyrLev(pyr,ind,k);
      dim1v     = sz(1,1)*sz(1,2);
      dim1h     = sz(2,1)*sz(2,2);
      dim1d     = sz(3,1)*sz(3,2);
      W.HP(k).V = reshape(lev(1:dim1v),sz(1,1),sz(1,2));
      W.HP(k).H = reshape(lev(dim1v+1:dim1v+dim1h),sz(2,1),sz(2,2));
      W.HP(k).D = reshape(lev(dim1v+dim1h+1:dim1v+dim1h+dim1d),sz(3,1),sz(3,2));
   end
      
   sz   = ind(length(ind),:);
   lev  = pyr(length(pyr)-sz(1)*sz(2)+1:length(pyr));
   W.LP = reshape(lev, sz(1)*sz(2),1); % lowpass subband
   return;
   
%%% ----------------------------------------------------------------
%%% Extract statistical feature vector
%%%
function [ftr] = stat_ftr(W)

   lev = length(W(1).HP);
   ftr = [];
   
   if length(W) == 1 % grayscale
      for k = 1:lev-1
	 VERT = get_neighbor(W,k,'v');
	 HORIZ = get_neighbor(W,k,'h');
	 DIAG = get_neighbor(W,k,'d');
	 
	 V = VERT(:,1);
	 H = HORIZ(:,1);
	 D = DIAG(:,1);
	 M1 = [mean(V) mean(H) mean(D)];
	 M2 = [var(V) var(H) var(D)];
	 M3 = [kurtosis(V) kurtosis(H) kurtosis(D)];
	 M4 = [skewness(V) skewness(H) skewness(D)];
	 
	 %%% Linear predictor of coefficient magnitude
	 V = abs(V);
	 nzind = find( V>=1 );
	 V = V(nzind);
	 Qv = abs(VERT(nzind,[2:8]));
	 v(:,k) = inv(Qv'*Qv) * Qv' * V;
	 
	 H = abs(H);
	 nzind = find( H>=1 );
	 H = H(nzind);
	 Qh = abs(HORIZ(nzind,[2:8]));
	 h(:,k) = inv(Qh'*Qh) * Qh' * H;
	 
	 D = abs(D);
	 nzind = find( D>=1 );
	 D = D(nzind);
	 Qd = abs(DIAG(nzind,[2:8]));
	 d(:,k) = inv(Qd'*Qd) * Qd' * D;
	 
	 %%% Difference between actual and predicted coefficients
	 Vp = Qv * v(:,k);
	 Hp = Qh * h(:,k);
	 Dp = Qd * d(:,k);
	 Ev = (log2(V) - log2(abs(Vp)));
	 Eh = (log2(H) - log2(abs(Hp)));
	 Ed = (log2(D) - log2(abs(Dp)));
	 
	 M5 = [mean(Ev) mean(Eh) mean(Ed)];
	 M6 = [var(Ev) var(Eh) var(Ed)];
	 M7 = [kurtosis(Ev) kurtosis(Eh) kurtosis(Ed)];
	 M8 = [skewness(Ev) skewness(Eh) skewness(Ed)];
	 
	 T1 = [M1 M2 M3 M4];
	 T2 = [M5 M6 M7 M8];
	 ftr = [ftr T1(:)' T2(:)'];
      end
   elseif length(W) == 3 % color
      for l = 0:2
	 for k = 1:lev-1
	    VERT = get_neighbor(W(l+1),k,'v');
	    HORIZ = get_neighbor(W(l+1),k,'h');
	    DIAG = get_neighbor(W(l+1),k,'d');
	    
	    v1 = get_neighbor(W(mod(l+1,3)+1),k,'v');
	    h1 = get_neighbor(W(mod(l+1,3)+1),k,'h');
	    d1 = get_neighbor(W(mod(l+1,3)+1),k,'d');
	    
	    v2 = get_neighbor(W(mod(l+2,3)+1),k,'v');
	    h2 = get_neighbor(W(mod(l+2,3)+1),k,'h');
	    d2 = get_neighbor(W(mod(l+2,3)+1),k,'d');	 
	    
	    V = VERT(:,1);
	    H = HORIZ(:,1);
	    D = DIAG(:,1);
	    
	    M1 = [mean(V) mean(H) mean(D)];
	    M2 = [var(V) var(H) var(D)];
	    M3 = [kurtosis(V) kurtosis(H) kurtosis(D)];
	    M4 = [skewness(V) skewness(H) skewness(D)];
	    
	    %%% Linear predictor of coefficient magnitude
	    VERT = [VERT v1(:,1) v2(:,1)];
	    HORIZ = [HORIZ h1(:,1) h2(:,1)];
	    DIAG = [DIAG d1(:,1) d2(:,1)];
	 
	    V = abs(V);
	    nzind = find( V>=1 );
	    V = V(nzind);
	    Qv = abs(VERT(nzind,[2:end]));
	    v(:,k) = inv(Qv'*Qv) * Qv' * V;
	    
	    H = abs(H);
	    nzind = find( H>=1 );
	    H = H(nzind);
	    Qh = abs(HORIZ(nzind,[2:end]));
	    h(:,k) = inv(Qh'*Qh) * Qh' * H;
	    
	    D = abs(D);
	    nzind = find( D>=1 );
	    D = D(nzind);
	    Qd = abs(DIAG(nzind,[2:end]));
	    d(:,k) = inv(Qd'*Qd) * Qd' * D;
	    
	    %%% Difference between actual and predicted coefficients
	    Vp = Qv * v(:,k);
	    Hp = Qh * h(:,k);
	    Dp = Qd * d(:,k);
	    Ev = (log2(V) - log2(abs(Vp)));
	    Eh = (log2(H) - log2(abs(Hp)));
	    Ed = (log2(D) - log2(abs(Dp)));
	    
	    M5 = [mean(Ev) mean(Eh) mean(Ed)];
	    M6 = [var(Ev) var(Eh) var(Ed)];
	    M7 = [kurtosis(Ev) kurtosis(Eh) kurtosis(Ed)];
	    M8 = [skewness(Ev) skewness(Eh) skewness(Ed)];
	    
	    T1 = [M1 M2 M3 M4];
	    T2 = [M5 M6 M7 M8];
	    ftr = [ftr T1(:)' T2(:)'];
	 end   
      end
   end
   return;
      
%%% ----------------------------------------------------------------
%%% Helper function for stat_ftr() -- extract spatial/scale/orientation
%%% neighbors
%%%
function [nb] = get_neighbor(W,lev,band)
   
   if lev >= length(W.HP)
      lev = num2str(length(W.HP)-1);
      error(['Up to ' lev 'levels are permitted']);
   end
   
   [ydim,xdim] = size(W.HP(lev).V);
   nb = zeros((xdim-2)*(ydim-2), 8);
   xlim = [2:xdim-1];
   ylim = [2:ydim-1];
   dim = prod((xdim-2)*(ydim-2));
   
   switch band
    case 'v',
     nb(:,1) = reshape(W.HP(lev).V(ylim,xlim), dim, 1);
     nb(:,2) = reshape(W.HP(lev).V(ylim-1,xlim), dim, 1);
     nb(:,3) = reshape(W.HP(lev).V(ylim,xlim-1), dim, 1);
     nb(:,4) = reshape(W.HP(lev+1).V(round(ylim/2), round(xlim/2)), dim, 1);
     nb(:,5) = reshape(W.HP(lev).D(ylim,xlim), dim, 1);
     nb(:,6) = reshape(W.HP(lev+1).D(round(ylim/2), round(xlim/2)), dim, 1);
     nb(:,7) = reshape(W.HP(lev).V(ylim+1,xlim), dim, 1);
     nb(:,8) = reshape(W.HP(lev).V(ylim,xlim+1), dim, 1);
    case 'h',
     nb(:,1) = reshape(W.HP(lev).H(ylim,xlim), dim, 1);
     nb(:,2) = reshape(W.HP(lev).H(ylim-1,xlim), dim, 1);
     nb(:,3) = reshape(W.HP(lev).H(ylim,xlim-1), dim, 1);
     nb(:,4) = reshape(W.HP(lev+1).H(round(ylim/2), round(xlim/2)), dim, 1);
     nb(:,5) = reshape(W.HP(lev).D(ylim,xlim), dim, 1);
     nb(:,6) = reshape(W.HP(lev+1).D(round(ylim/2), round(xlim/2)), dim, 1);
     nb(:,7) = reshape(W.HP(lev).H(ylim+1,xlim), dim, 1);
     nb(:,8) = reshape(W.HP(lev).H(ylim,xlim+1), dim, 1);
    case 'd',
     nb(:,1) = reshape(W.HP(lev).D(ylim,xlim), dim, 1);
     nb(:,2) = reshape(W.HP(lev).D(ylim-1,xlim), dim, 1);
     nb(:,3) = reshape(W.HP(lev).D(ylim,xlim-1), dim, 1);
     nb(:,4) = reshape(W.HP(lev+1).D(round(ylim/2), round(xlim/2)), dim, 1);
     nb(:,5) = reshape(W.HP(lev).H(ylim,xlim), dim, 1);
     nb(:,6) = reshape(W.HP(lev).V(ylim,xlim), dim, 1);
     nb(:,7) = reshape(W.HP(lev).D(ylim+1,xlim), dim, 1);
     nb(:,8) = reshape(W.HP(lev).D(ylim,xlim+1), dim, 1);
    otherwise,
     error('Bad subband label');
   end
   return;

%%% ----------------------------------------------------------------
%%% Crop central region of size w x h
%%%
function [im1] = im_center_crop(im,w,h)

   if( size(im,1) > size(im,2) )
      b = center_blk([size(im,1),size(im,2)],[max(h,w) min(h,w)]);
   else
      b = center_blk([size(im,1),size(im,2)],[min(h,w) max(h,w)]);
   end
   im1 = imcrop(im,b);
   return;
   
function [b] = center_blk(B,bs)
   B = B(:)';
   bs = bs(:)';
   
   if length(B) == 2
      B = [1 1 B(1) B(2)];
   end
   if length(B) ~= 4
      error('Specify the block please');
   end
   if length(bs) ~= 2
      error('Specify the width and height please');
   end
   
   ctry = round((B(3)-B(1))/2);
   ctrx = round((B(4)-B(2))/2);
   w    = ceil(bs(2)/2);
   h    = ceil(bs(1)/2);
   b    = [ max(1,ctrx-w) max(1,ctry-h) bs(2)-1 bs(1)-1 ];
   return;
   
%%% ----------------------------------------------------------------

