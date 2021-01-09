script
clc
clear
disp('% �� �16. ����������������� ������������ ������')
disp('%')
disp('%')
disp('% ������� �������� ������')
DATA=0;
while DATA==0
Nb = input('Nb = ');      % ����� �������
N = 128;        % ����� ������������������
Fs = 1000*(mod(Nb,5)+1);      % ������� ������������� (��)
A1 = 0.8+0.01*Nb;      % ��������� ���������� ��������
A2 = 1.5*A1;   
f1 = Fs/8;      % ������� ���������� �������� (��)
f2 = 2*f1;    
sigma = [0, A1, 2*A1, 4*A1];% ������ �������� ��� ����
disp('% ��������� ������������ ����� �������� ������')
disp('% ��� ���������� �������� ������ ������� 1')
disp('% ��� ������������ �������� ������ ������� 0 � ��������� ����')
DATA = input('--> ');
end
disp('%')
disp('%')
disp('% ��� ����������� ������� <ENTER>')
pause
disp('%')
disp('%')
disp('% �.1. �������� ��������������� ������������� � ����������� �� ������ ����')
disp('%')
disp('%')
disp('% ��� ������ �������� ������������������� � ������������')
disp('% � ���������� ��� ���� ������� <ENTER>')
pause
w1 = 2*pi*f1/Fs; w2 = 2*pi*f2/Fs;  % ������������� ������� ���������� �������� (���)
n = 0:(N-1);                       % ���������� ������������� �����
x = A1*cos(w1*n')+A2*cos(w2*n');   % ������������������ � ����� ���� �������� (������-�������)
figure('Name',' Harmonic Signals Embedded in White Gaussian Noise and Periodograms','NumberTitle', 'off')
for i = 1:length(sigma)         % ������ �������� ������� sigma
xe = x'+sigma(i).*randn(1,N);   % ������������������ � ��������� ��� ���� (x'-������-������)
subplot(4,2,2*i-1), plot(n,xe,'Linewidth',2), grid
xlabel('n'), ylabel(strcat('xe',num2str(i),'(n)'))
title(strcat(['Sequence ',num2str(i), '   STD = ',num2str(sigma(i))]))
[Se,f] = periodogram(xe,[],N,Fs,'twosided'); % ������������� ������������������ � �������� ��� ����
subplot(4,2,2*i), plot(f,Se,'Linewidth',2), grid
xlabel('f (Hz)'), ylabel(strcat('S',num2str(i),'(f)'))
title(strcat(['Periodogram ',num2str(i),' (Vt/Hz)   STD = ',num2str(sigma(i))]))
end
disp('%')
disp('%')
disp('% ��� ����������� ������� <ENTER>')
pause
disp('%')
disp('%')
disp('% �.2. �������� ��������������� ������������� � �����������')
disp('% �� ������� ������������� �� �������')
disp('%')
disp('%')
disp('% ��� ������ �������� ������������ ��� ��������� ����������� ��� ������� <ENTER>')
pause
figure('Name','Periodograms of Harmonic Signal with defferent DFT size','NumberTitle', 'off')
xe = x'+randn(1,N);         % ������������������ ����� N
M = [N/16 N/8 N 8*N];       % ������ ������������ ���
for i = 1:length(M)
[Se,f] = periodogram(xe,[],M(i),Fs,'onesided');   % ������������� � �������� ������ ������ ��� �������� ����������� ���  
subplot(4,1,i), plot(f, Se,'Linewidth',2),grid
xlabel('f (Hz)'), ylabel(strcat('S',num2str(i),'(f)'))
title(strcat(['Periodogram ',num2str(i),' (Vt/Hz)   length DFT=', num2str(M(i))]))
end
disp('%')
disp('%')
disp('% ��� ����������� ������� <ENTER>')
pause
disp('%')
disp('%')
disp('% �.3. �������� ������ ��� �� ��������������� ������������� � ���������������')
disp('%')
disp('%')
disp('% ��� ������ �������� ����������� �������� � �������� ��������')
disp('% ���������� �������� ��� �� �� ������ ������� <ENTER>')
pause
SWN = var(randn(1,100000))/Fs;        % �������� ��� ����������� ������ ����
N_WN = 1000:1000:100000;              % ������ ���� ����
for i = 1:length(N_WN)                % ������ �������� ������� N_WN
e = randn(1,N_WN(i));                 % ��� �������� �����
SWN_estimate = periodogram(e,[],N_WN(i),Fs,'twosided');   % ������ ��� ����
beta(i) = mean(SWN-SWN_estimate');    % ��������� ������ ���
mean_square(i) = var(SWN_estimate)+beta(i)^2;            % ������� ������� ���������� �������� ��� �� �� ������
end
figure('Name','Bias and Mean square Deviation of true PSD from its Estimate','NumberTitle', 'off')
subplot(2,1,1), plot(N_WN, beta,'LineWidth',2), grid, xlabel('N')
ylabel('beta')
title('Bias')
subplot(2,1,2), plot(N_WN, mean_square,'LineWidth',2), grid, xlabel('N')
ylabel('meanerr')
title('Mean square Deviation of true PSD from its Estimate')
disp('%')
disp('%')
disp('% ��� ����������� ������� <ENTER>')
pause
disp('%')
disp('%')
disp('% �.4. ������������ ��������� ������������������ � ��������� ���')
N = 1000;                        % ����� ����
xw = randn(1,N);                 % ���������� ����� ��� ����� N
N02 = var(xw)+(mean(xw)).^2;     % ��������� N0/2
m = -(N-1):(N-1);                % ���������� ������������� ����� ��� ���, �������������� ������������ m = 0
Ry_required = 0.25.*0.95.^abs(m); % ��������� ���
L = 2*N-1;                        % ����� ���
m = 0:L-1;                        % ���������� ������������� ����� ��� ���, �������������� ������������ m = N
disp('%')
disp('%')
disp('% ��� ������ ������� ��������� ��� ������� <ENTER>')
pause
figure('Name','Required ACF','NumberTitle', 'off')
stem(m,Ry_required), grid, xlabel('m'), title('Required ACF Ry')
disp('%')
disp('%')
disp('% � ������� ������ Zoom in ���������� � ������� ������ ������� ���-�������')
DATA=0;
while DATA==0
disp('%')
R = input('    R = ');            % ������ ������� ���-�������
disp('%')
disp('% ��������� ������������ ����� �������� ������')
disp('% ��� ���������� �������� ������ ������� 1')
disp('% ��� ������������ �������� ������ ������� 0 � ��������� ����')
DATA = input('--> ');
end
Sy = 2*real(fft(Ry_required(N:L),L))-Ry_required (N); % ��� � L ������, ����������� �� ��������� ���
A = sqrt(real(Sy)./N02);          % ��� ���-������� � L ������
k = 0:L-1;                        % ���������� ������������� �������
F = -k*pi*R/L;                    % ���� ���-������� � L ������
j = sqrt(-1);                  
H = A.*exp(j*F);                  % �� ���-������� � L ������
h1 = real(ifft(H));               % �� ���-������� �� ������� L �������� �������
y_ACF = fftfilt(h1,xw);           % ��������� ������������������ ����� N � ��������� ��� �� ������ ���-�������
h = h1(1:R+1);                    % �� ����� R+1
Ry_estimate = xcorr(y_ACF)./N;    % ������ ��� ������� ���-�������, �������������� ������������ N
disp('%')
disp('%')
disp('% ��� ������ ������� ��������� ��� � ������ ��� ������� ������� <ENTER>')
pause
figure('Name','Required ACF and ACF Estimate of Output Signal','NumberTitle', 'off')
subplot(2,1,1), stem(m,Ry_required), grid, title('Required ACF Ry')
subplot(2,1,2), stem(m,Ry_estimate), grid, xlabel('m')
title(' ACF Estimate of Output Signal - Ry estimate')
disp('%')
disp('%')
disp('% ��� ������ �������� ��, ����������� ������ ���� � ������� ���-������� ������� <ENTER>')
pause
n = 0:(N-1);  % ���������� ������������� ����� ��� ����������� � ������� ���-�������
figure('Name','Impulse Response, Input and Output Signals','NumberTitle', 'off')
subplot(3,1,1), stem(0:R,h), grid, title('Impulse Response h(n)')
subplot(3,1,2), plot(n,xw), grid
title('Input Signal - White Gaussian Noise')
subplot(3,1,3), plot(n,y_ACF), grid, xlabel('n')
title('Output Signal with Required ACF - y ACF')
disp('%')
disp('%')
disp('% ��� ����������� ������� <ENTER>')
pause
disp('%')
disp('%')
disp('% �.5. ���������� ��������� ������������������ � ��������� ���')
disp('%')
disp('%')
disp('% ��� ������ �������� �������� � ��������������� ������������������� ������� <ENTER>')
pause
[b,a] = butter(3,0.3,'high');      % ������������ ���-������� ��� �����������
y = filter(b,a,y_ACF);             % ������������������ �� ������ ���-������� ����� �������� ������
figure('Name','Input and Output Signals of Butterworth filter','NumberTitle', 'off')
subplot(2,1,1), plot(n,y_ACF), grid, title('Input Signal of Butterworth filter � y ACF')
subplot(2,1,2), plot(n,y), grid, xlabel('n'), ylabel('y(n)')
title('Output Signal of Butterworth filter - y')
disp('%')
disp('%')
disp('% ��� ����������� ������� <ENTER>')
pause
disp('%')
disp('%')
disp('% �.6. ������ �������������')
disp('%')
disp('%')
disp('% ��� ������ ������� ������������� ������� <ENTER>')
pause
[S,f] = periodogram(y,[],N,Fs,'twosided');         % ������������� ��������� ������������������
figure('Name','Periodogram of the Non-white Gaussian Noise','NumberTitle', 'off')
subplot(2,1,1), plot(f,S,'Linewidth',2), grid
xlabel('f (Hz)'), ylabel('S(f) (Vt/Hz)')
title('Periodogram of the Non-white Gaussian noise')
subplot(2,1,2), periodogram(y,[],N,Fs,'twosided')
xlabel('f (Hz)'), ylabel('S(f) (dB/Hz)')
title('Periodogram of the Non-white Gaussian noise')
disp('%')
disp('%')
disp('% ��� ����������� ������� <ENTER>')
pause
disp('%')
disp('%')
disp('% �.7. ������ ������������� ��������')
disp('%')
disp('%')
disp('% ��� ������ �������� ������������ ��������� ������� <ENTER>')
pause
K = [5 10 20];                   % ������ ���������� ����������� ������
figure('Name','Daniell Periodograms for the Different Number of Frequency Intervals','NumberTitle', 'off')
for i = 1:3                      % ������ �������� ������� K
S1 = [S(N-K(i)+1:N); S; S(1:K(i))];  % �������������, ������������ ������������ ����� � ������ �� K ��������
S2 = smooth(S1,K(i));            % ��������� ���������� ����������� �������� (������ ����� N+2K(i))
SD(:,i) = S2(K(i)+1:N+K(i));     % ������������� ��������� (������ ����� N) ��� ���������� ����������� ������ K(i)
subplot(4,1,i+1), plot(f, SD(:,i),'Linewidth',2), grid
xlabel('f (Hz)'), ylabel(strcat('SD',num2str(i),'(f)'))
title(strcat(['Daniell Periodogram ',num2str(i),'  Frequency Interval 2K+1, K=',num2str(K(i))]))
end
subplot(4,1,1)
plot(f,S,'Linewidth',2), grid, xlabel('f (Hz)'), ylabel('S(f)')
title('Original non-modified periodogram (Vt/Hz)')
disp('%')
disp('%')
disp('% ��� ����������� ������� <ENTER>')
pause
disp('%')
disp('%')
disp('% �.8. ������ ������������� ���������')
disp('%')
disp('%')
disp('% ��� ������ �������� ������������� ��������� ������� <ENTER>')
pause
L = [10 20 40];                         % ������ ���� ����������
figure('Name',' Bartlett Periodograms for Different Fragment Lengths','NumberTitle', 'off')
for i = 1:3                             % ������ �������� ������� L
SB(:,i) = pwelch(y,rectwin(L(i)),0,N,Fs,'twosided'); % ������������� ��������� ��� ��������� ����� L(i)
subplot(4,1,i+1), plot(f, SB(:,i),'Linewidth',2), grid
xlabel('f (Hz)'), ylabel(strcat('SB',num2str(i),'(f)'))
title(strcat([' Bartlett Periodogram ',num2str(i),'  L =',num2str(L(i))]))
end
subplot(4,1,1)
plot(f,S,'Linewidth',2), grid, xlabel('f (Hz)'), ylabel('S(f)')
title('Original non-modified periodogram (Vt/Hz)')
disp('%')
disp('%')
disp('% ��� ����������� ������� <ENTER>')
pause
disp('%')
disp('%')
disp('% �.9. ������ ������������� �����')
disp('%')
disp('%')
disp('% ��� ������ �������� ������������� ����� ������� <ENTER>')
pause
num = 1;                      % ���������� ����� ������������� �����
L = ceil([0.1 0.05 0.025].*length(y));      % ������ ���� ����������
figure('Name','Welch Periodograms for Different Fragment Lengths and Overlapping','NumberTitle', 'off')
for i = 1:3                              % ������ �������� ������� L
Q = ceil(0.0125*length(y));              % �������� ����������
SW(:,num) = pwelch(y,L(i),Q,N,Fs,'twosided'); % ������������� ����� ��� ����� ��������� L(i) � �������� ���������� Q
subplot(4,1,i+1), plot(f,SW(:,num),'Linewidth',2), grid
xlabel('f (Hz)'), ylabel('S(f)')
title(strcat(['Welch Periodogram Fragment length L = ',num2str(L(i)), '  Overlapping Q = ',num2str(Q)]))
num = num+1;
subplot(4,1,1), plot(f,S,'Linewidth',2), grid, xlabel('f (Hz)')
ylabel('S(f)'), title('Original non-modified periodogram (Vt/Hz)')
end
disp('%')
disp('%')
disp('% ��� ����������� ������� <ENTER>')
pause
disp('%')
disp('%')
disp('% �.10. ������ ������ ��� �� ������ ��������-�����')
disp('%')
disp('%')
disp('% ��� ������ �������� ������ ���, ����������� �� ������ ��������-�����, ������� <ENTER>')
pause
N1 = ceil(N/10);               % ������������ ����� �� ������� ��� ������ ��� ���������������� ����
R1 = (1/N).*xcorr(y,N1);       % ������ ��� ���������������� ���� ����� 2N1+1, �������������� ������������ N1+1
R = R1(2:length(R1)-1);        % ������ ��� ���������������� ���� ����� 2N1-1, �������������� ������������ N1
L1 = length(R);                % ����� ������ ���
Rw(:,1) = R'.*rectwin(L1);     % ���, ���������� ������������� �����
Rw(:,2) = R'.*hamming(L1);     % ���, ���������� ����� ��������
Rw(:,3) = R'.*chebwin(L1);     % ���, ���������� ����� ��������
name(1).win = 'Rectangular Window';   % ����� ���� (������ ������� name � ����� ����� win)
name(2).win = 'Hamming Window';
name(3).win = 'Chebyshev Window';
f = 0:Fs/N:Fs-Fs/N;             % ������ ������ (��)
figure('Name','PSD estimates by the Blackman-Tukey method','NumberTitle', 'off')
for i = 1:3                     % ������ ����
SBT(:,i) = (1/Fs)*(2*real(fft(Rw(N1:L1,i),N))- Rw(N1,i));  % ������ ��� ����� N, ����������� �� ���, ���������� �����
subplot(4,1,i+1), plot(f,SBT(:,i),'Linewidth',2), grid
xlabel('f (Hz)'), ylabel(strcat('SBT',num2str(i),'(f)'))
title(['PSD Estimate  -  ',strcat(name(i).win)])
end
subplot(4,1,1)
plot(f,S,'Linewidth',2), grid, xlabel('f (Hz)'), ylabel('S(f)')
title('Original non-modified periodogram (Vt/Hz)')
disp('%')
disp('%')
disp('% ��� ����������� ������� <ENTER>')
pause
disp('%')
disp('%')
disp('% �.11. ����������� ����������� �������� ������ ���')
disp('%')
disp('%')
disp('% ��� ������ �������� ��� ������� <ENTER>')
pause
disp('%')
disp('   ��� �������������')
format long                   % ������� ������ ��� ������ ���
STD_S = std(S)                % ��� �������������
disp('   ��� ������������ �������� ��� ��������� ���������� ����������� ������ K')
STD_SD = [K' std(SD)']        % ���������� ����������� ������ � ��� ������������ ��������
disp('   ��� ������������ ��������� ��� ��������� ����� ��������� L')
STD_SB = [L' std(SB)']        % ����� ���������� � ��� ������������ ���������
disp('   ��� ������������� ����� ��� ��������� ����� ��������� L � �������� ���������� Q')
LL = [L(1) L(2) L(3)];        % ����� ����������
Q = [Q Q Q];                  % �������� ����������
STD_SW = [LL' Q' std(SW)']    % ����� ����������, �������� ���������� � ��� ������������ �����
disp('   ��� ������ ��� �� ������ ��������-����� ��� ��������� �����')
WINDOW = {name.win};          % ����� ���� (������ ����� � cell array)
STD_SBT = std(SBT);           % ��� ������ ��� �� ������ ��������-�����
STD_SBT = [WINDOW(1)' STD_SBT(1)'; WINDOW(2)' STD_SBT(2)'; WINDOW(3)' STD_SBT(3)']    % ����� ���� � ��� ������ ��� �� ������ ��������-�����
disp('%')
disp('%')
disp('% ��� ������ �������� ����������� ������� <ENTER>')
pause
disp('%')
format          % ������ � ��������� ������� ��� ������ ������������
disp('   ����������� �������������')
Q_S = mean(S).^2/var(S)               % ����������� �������������
disp('   ����������� ������������ �������� ��� ��������� ���������� ����������� ������ K')
Q_SD = mean(SD).^2./var(SD);          % ����������� ������������ ��������
Q_SD = [K' Q_SD']
disp('   ����������� ������������ ��������� ��� ��������� ����� ��������� L')
Q_SB = mean(SB).^2./var(SB);          % ����������� ������������ ���������
Q_SB = [L' Q_SB']
disp('   ����������� ������������ ����� ��� ��������� ����� ��������� L � �������� ���������� Q')
Q_SW = mean(SW).^2./var(SW);          % ����������� ������������ �����
Q_SW = [LL' Q' Q_SW']
disp('   ����������� ������ ��� �� ������ ��������-����� ��� ��������� �����')
Q_SBT = real(mean(SBT).^2./var(SBT)); % ����������� ������ ��� �� ������ ��������-�����
Q_SBT = [WINDOW(1)' Q_SBT(1)'; WINDOW(2)' Q_SBT(2)'; WINDOW(3)' Q_SBT(3)']
disp('% ��� ����������� ������� <ENTER>')
pause
disp('%')
disp('%')
disp('% �.12. ���������� �������������')
disp('%')
disp('%')
disp('% ��� ������ ������� ������������� ����������� �������������� ������� ������� <ENTER>')
pause
N = 4000;                          % ����� ������������������
n = 0:N-1;                         % ���������� ������������� �����
x = A1*cos(w1*n')+A2*cos(w2*n');   % ������������������ � ����� ���� ��������
figure('Name','Harmonic Signal Spectrogram','NumberTitle', 'off')
spectrogram(x,128,120,128,Fs,'yaxis')
colorbar
xlabel('Time (s)'), ylabel('Frequency (Hz)')
title('Harmonic Signal Spectrogram')
disp('%')
disp('%')
disp('% ������ ���������')