clc;close all; clear all;
data = cell(1,50);
gn=[0:49];
for i = 1:50
    g = ['report-def-' num2str(gn(i))  '-rfile.out'] ;
    data(1,i) = {textread(g,'','headerlines',3)};
end
for i = 1:50
    f = data{1,i};
    pp(:,i) = f(:,2);      
end
%%
s=[-0.7 0.75 -0.5];

%% training data fft
train_pp=pp;
sizetrpp=size(train_pp);
fs=1/10^-5;
n = length(pp(8752:18752,1)); 
for i=1:sizetrpp(2)
    y=fft(train_pp(8752:18752,i));
    fshift = fs*(0:n/2)/n;
    yshift = fftshift(y);
    yy=y(1:n/2+1);
    train_angle2(:,i)=yy*2/n;
end
%%
figure
hold on
plot(0.08751:0.00001:0.18751,pp(8752:18752,1),'k-')
title('Signal from ANSYS fluent')
xlabel('Time (sec)')
ylabel('Pressure (Pa)')
%%
figure
hold on
plot(fshift(1:100),real(train_angle2((1:100),i)),'k-',fshift(1:100),imag(train_angle2((1:100),i)),'k--')
title('Frequency domain signal')
legend('Real number','Imaginary number')
xlabel('Frequency (Hz)')
ylabel('Amplitude')
%%
for i=1:5001
    abb(i)=sqrt(real(train_angle2(i,end))^2+imag(train_angle2(i,end))^2);
end
figure
hold on
plot(fshift(1:1001),abb(1:1001),'k-')
title('Frequency domain signal')
xlabel('Frequency (Hz)')
ylabel('Amplitude')
%%
train_angle= train_angle2(1:100,:);
%%
input_data=[];
output_data=[];
n=0;
for i=1:50
    input_data=[input_data;train_angle(:,i);];
end
output_data=s';

%%
fid = fopen('tpin070075n050.txt','w');
size_final=size(input_data);
for i=1:size_final(1)
     for j=1:size_final(2) 
        fprintf(fid,'%.10f\t',real(input_data(i,j)));   
     end
     fprintf(fid,'\r\n');  
end
fid = fopen('tpon070075n050.txt','w');
size_final=size(output_data);
for i=1:size_final(1)
     for j=1:size_final(2) 
        fprintf(fid,'%.10f\t',output_data(i,j));   
     end
     fprintf(fid,'\r\n');  
end
fid = fopen('tpi2n070075n050.txt','w');
size_final=size(input_data);
for i=1:size_final(1)
     for j=1:size_final(2) 
        fprintf(fid,'%.10f\t',imag(input_data(i,j)));   
     end
     fprintf(fid,'\r\n');  
end
