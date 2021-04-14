clc
clear all
close all


s = 2.4;  % object size unit:mm
pitch1 = 0.001; % sampling interval of the object unit:mm
n1 = s/pitch1;
[x1,y1] = meshgrid(linspace(-s/2,s/2-pitch1,n1),linspace(-s/2,s/2-pitch1,n1));
lam = 500e-6; % wavelength unit:mm
k = 2*pi/lam;

t = ones(n1,n1);
%% at center (firs use this case and then comment this case and uncomment the next part)
t(n1/2-10:n1/2+10,n1/2-5) = 0;
t(n1/2-10:n1/2+10,n1/2-4) = 0;
t(n1/2-10:n1/2+10,n1/2-1) = 0;
t(n1/2-10:n1/2+10,n1/2) = 0;
t(n1/2-10:n1/2+10,n1/2+3) = 0;
t(n1/2-10:n1/2+10,n1/2+4) = 0;

%% at x=0.9 mm
% t(n1/2-10:n1/2+10,n1/8*7-5) = 0;
% t(n1/2-10:n1/2+10,n1/8*7-4) = 0;
% t(n1/2-10:n1/2+10,n1/8*7-1) = 0;
% t(n1/2-10:n1/2+10,n1/8*7) = 0;
% t(n1/2-10:n1/2+10,n1/8*7+3) = 0;
% t(n1/2-10:n1/2+10,n1/8*7+4) = 0;

figure,imshow(t,[])

f = 20;     % source-object distance unit:mm
R = sqrt(x1.^2+y1.^2+f^2);
t = t.*exp(1i*k*R);

z = 60; % object-detector distance unit:mm
S = 10; % full detector size unit:mm

L = (s+S)/2;

pitch2 = 0.005; %pixel pitch of the detector unit:mm
n2 = S/pitch2;

[X,Y] = meshgrid(linspace(-S/2,S/2-pitch2,n2),linspace(-S/2,S/2-pitch2,n2));

%% hologram generation
% see our paper Generalized diffraction calculation method for holography and diffractive optics
fmax = min(1/2/pitch1,L/lam/sqrt(z^2+L^2));
fxf = [-fmax,0,fmax];
pitch_f = 1/3/L;
M = (floor(2*fmax/pitch_f)+mod(floor(2*fmax/pitch_f),2));

[fxn,fyn] = meshgrid(linspace(-M/2*pitch_f,M/2*pitch_f-pitch_f,M),linspace(-M/2*pitch_f,M/2*pitch_f-pitch_f,M));
Hn = exp(1i*k*(z*sqrt(1-(fxn*lam).^2-(fyn*lam).^2)));

xx = x1(:);
yy = y1(:);
fxk = fxn(:);
fyk = fyn(:);
X1 = X(:);
Y1 = Y(:);

iflag = -1;
eps = 10^(-10);
K = n1/2/(1/2/pitch1);
rr = S/s;

tic
t_asmNUFT = nufft2d3(n1^2,xx/max(abs(xx))*pi,yy/max(abs(yy))*pi,t,iflag,eps,M^2,fxk*K,fyk*K);
t_asmNUFT = reshape(t_asmNUFT,[M,M]);
t_pro_asmNUFT = nufft2d3(M^2,fxk*K*rr,fyk*K*rr,Hn.*t_asmNUFT,-iflag,eps,n2^2,X1/max(abs(X1))*pi,Y1/max(abs(Y1))*pi);
toc


t_pro_asmNUFT = t_pro_asmNUFT/max(abs(t_pro_asmNUFT));
t_pro_asmNUFT = reshape(t_pro_asmNUFT,[n2,n2]);

amplitude_asm_ex = abs(t_pro_asmNUFT);
intensity_ex = amplitude_asm_ex.^2;

figure,imshow(intensity_ex,[]);axis equal;
title('hologram')

%% reconstruction

pp = intensity_ex;
% pp(:,1:n2/8) = 0; % cut off the left one-eighth part 
figure,imshow(pp,[])
n3 = length(pp);

mm = (z+f)/f;
pitchn = pitch2/mm;
zn = -z/mm;
[fxr,fyr] = meshgrid(linspace(-1/2/pitchn,1/2/pitchn-1/n2/pitchn,n3)',linspace(-1/2/pitchn,1/2/pitchn-1/n2/pitchn,n3)');
Hr = exp(1i*k*zn*sqrt(1-(fxr*lam).^2-(fyr*lam).^2));
aa = fftshift(fft2(fftshift(pp)));
rec = ifftshift(ifft2(ifftshift(padarray(aa.*Hr,[n2/2,n2/2]))));
rec = rec./max(abs(rec(:)));
figure,imshow(abs(rec),[])


