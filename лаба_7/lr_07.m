script
clc
clear
disp('% ЛР №7. ДИСКРЕТНЫЕ СИГНАЛЫ')
disp('%')
disp('%')
disp('% Введите ИСХОДНЫЕ ДАННЫЕ');
DATA=0;
while DATA==0
Nb = input('Nb = ');         % НОМЕР БРИГАДЫ
N = 30 + mod(Nb,5);          % ДЛИНА ПОСЛЕДОВАТЕЛЬНОСТИ
T = 0.0005*(1+mod(Nb,3));    % ПЕРИОД ДИСКРЕТИЗАЦИИ
a = (-1)^Nb*(0.8+0.005*Nb);  % ОСНОВАНИЕ ДИСКРЕТНОЙ ЭКСПОНЕНТЫ
C = 1+mod(Nb,5);             % АМПЛИТУДА ДИСКРЕТНОГО ГАРМОНИЧЕСКОГО СИГНАЛА
w0 = pi()/(6+mod(Nb,5));     % ЧАСТОТА ДИСКРЕТНОГО ГАРМОНИЧЕСКОГО СИГНАЛА
m = 5+mod(Nb,5);             % ВЕЛИЧИНА ЗАДЕРЖКИ
U = Nb;                      % АМПЛИТУДА ИМПУЛЬСА
n0 = mod(Nb,5)+3;            % МОМЕНТ НАЧАЛА ИМПУЛЬСА
n_imp = mod(Nb,5)+5;         % ДЛИТЕЛЬНОСТЬ ИМПУЛЬСА
B = [1.5+mod(Nb,5), 5.7-mod(Nb,5), 2.2+mod(Nb,5)];                  % ВЕКТОР АМПЛИТУД
w = [pi()/(4+mod(Nb,5)), pi()/(8+mod(Nb,5)), pi()/(16+mod(Nb,5))];  % ВЕКТОР ЧАСТОТ 
A = [1.5-mod(Nb,5), 0.7+mod(Nb,5), 1.4+mod(Nb,5)];                  % ВЕКТОР КОЭФФИЦИЕНТОВ ЛИНЕЙНОЙ КОМБИНАЦИИ
Mean = mod(Nb,5)+3;                  % ЗАДАННОЕ МАТЕМАТИЧЕСКОЕ ОЖИДАНИЕ ШУМА
Var = mod(Nb,5)+5;                   % ЗАДАННАЯ ДИСПЕРСИЯ ШУМА 
%N = input('N = ');              % ДЛИНА ПОСЛЕДОВАТЕЛЬНОСТИ
%T = input('T = ');              % ПЕРИОД ДИСКРЕТИЗАЦИИ
%a = input('a = ');              % ОСНОВАНИЕ ДИСКРЕТНОЙ ЭКСПОНЕНТЫ
%C = input('C = ');      % АМПЛИТУДА ДИСКРЕТНОГО ГАРМОНИЧЕСКОГО СИГНАЛА
%w0 = input('w0 = ');    % ЧАСТОТА ДИСКРЕТНОГО ГАРМОНИЧЕСКОГО СИГНАЛА
%m = input('m = ');              % ВЕЛИЧИНА ЗАДЕРЖКИ
%U = input('U = ');              % АМПЛИТУДА ИМПУЛЬСА
%n0 = input('n0 = ');            % МОМЕНТ НАЧАЛА ИМПУЛЬСА
%n_imp = input('n_imp = ');      % ДЛИТЕЛЬНОСТЬ ИМПУЛЬСА
%B = input('B = ');              % ВЕКТОР АМПЛИТУД
%w = input('w = ');              % ВЕКТОР ЧАСТОТ 
%A = input('A = ');        % ВЕКТОР КОЭФФИЦИЕНТОВ ЛИНЕЙНОЙ КОМБИНАЦИИ
%Mean = input('Mean = ');  % ЗАДАННОЕ МАТЕМАТИЧЕСКОЕ ОЖИДАНИЕ ШУМА
%Var = input('Var = ');    % ЗАДАННАЯ ДИСПЕРСИЯ ШУМА 
disp('% Проверьте ПРАВИЛЬНОСТЬ ввода ИСХОДНЫХ ДАННЫХ')
disp('% При ПРАВИЛЬНЫХ ИСХОДНЫХ ДАННЫХ введите 1')
disp('% При НЕПРАВИЛЬНЫХ ИСХОДНЫХ ДАННЫХ введите 0 и ПОВТОРИТЕ ввод')
DATA = input('--> '); 
end
disp('%')
disp('%')
disp('% Для продолжения нажмите <ENTER>')
pause 
disp('%')
disp('%')
disp('% п.1. ЦИФРОВОЙ ЕДИНИЧНЫЙ ИМПУЛЬС')
disp('%')
disp('%')
disp('% Для вывода ГРАФИКОВ цифрового единичного импульса нажмите <ENTER>')
pause 
n = 0:(N-1); nT = T.*n;      % ДИСКРЕТНОЕ НОРМИРОВАННОЕ И НЕНОРМИРОВАННОЕ ВРЕМЯ
u0 = [1 zeros(1,(N-1))];     % ЦИФРОВОЙ ЕДИНИЧНЫЙ ИМПУЛЬС
figure('Name','Digital Unit Impulse, Unit Step, and Discrete Exponent','NumberTitle', 'off')
subplot(3,2,1),stem(nT,u0,'Linewidth',2), grid
title('Digital Unit Impulse u0(nT)')
subplot(3,2,2),stem(n,u0,'Linewidth',2), grid 
title('Digital Unit Impulse u0(n)')
disp('%')
disp('%')
disp('% Для продолжения нажмите <ENTER>')
pause 
disp('%')
disp('%')
disp('% п.2. ЦИФРОВОЙ ЕДИНИЧНЫЙ СКАЧОК');
disp('%')
disp('%')
disp('% Для вывода ГРАФИКОВ цифрового единичного скачка нажмите <ENTER>')
pause 
u1 = [1 ones(1,(N-1))];       % ЦИФРОВОЙ ЕДИНИЧНЫЙ СКАЧОК
subplot(3,2,3),stem(nT,u1,'Linewidth',2), grid
title('Digital Unit Step u1(nT)'), 
subplot(3,2,4),stem(n,u1,'Linewidth',2), grid
title('Digital Unit Step u1(n)')
disp('%')
disp('%')
disp('% Для продолжения нажмите <ENTER>')
pause 
disp('%')
disp('%')
disp('% п.3. ДИСКРЕТНАЯ ЭКСПОНЕНТА')
disp('%')
disp('%')
disp('% Для вывода ГРАФИКОВ дискретной экспоненты нажмите <ENTER>')
pause
x1 = a.^n;                   % ДИСКРЕТНАЯ ЭКСПОНЕНТА
subplot(3,2,5),stem(nT,x1,'Linewidth',2), xlabel('nT'), grid
title('Discrete Exponent x1(nT)')
subplot(3,2,6),stem(n, x1,'Linewidth',2), xlabel('n'), grid
title('Discrete Exponent x1(n)'),
disp('%')
disp('%')
disp('% Для продолжения нажмите <ENTER>')
pause 
disp('%')
disp('%')
disp('% п.4. ДИСКРЕТНЫЙ КОМПЛЕКСНЫЙ ГАРМОНИЧЕСКИЙ СИГНАЛ')
disp('%')
disp('%')
disp('% Для вывода ГРАФИКОВ вещественной и мнимой частей')
disp('% гармонического сигнала нажмите <ENTER>')
pause 
x2 = C.*exp(j*w0.*n);  % ДИСКРЕТНЫЙ КОМПЛЕКСНЫЙ ГАРМОНИЧЕСКИЙ СИГНАЛ
figure('Name','Discrete Harmonic Signal','NumberTitle', 'off')
subplot(2,1,1),stem(n,real(x2) ,'Linewidth',2), grid
title('Discrete Harmonic Signal: REAL [x2(n)]')
subplot(2,1,2),stem(n,imag(x2) ,'Linewidth',2), xlabel('n'), grid
title(' Discrete Harmonic Signal: IMAG [x2(n)]')
disp('%')
disp('%')
disp('% Для продолжения нажмите <ENTER>')
pause 
disp('%')
disp('%')
disp('% п.5. ЗАДЕРЖАННЫЕ ПОСЛЕДОВАТЕЛЬНОСТИ')
disp('%')
disp('%')
disp('% Для вывода ГРАФИКОВ задержанных последовательностей нажмите <ENTER>')
pause
u0_m = [zeros(1,m) u0(1:(N-m))];    % ЗАДЕРЖАННЫЙ ЦИФРОВОЙ ЕДИНИЧНЫЙ ИМПУЛЬС
u1_m = [zeros(1,m) u1(1:(N-m))];    % ЗАДЕРЖАННЫЙ ЦИФРОВОЙ ЕДИНИЧНЫЙ СКАЧОК
x1_m = [zeros(1,m) x1(1:(N-m))];    % ЗАДЕРЖАННАЯ ДИСКРЕТНАЯ ЭКСПОНЕНТА
figure('Name','Delayed Discrete Signals','NumberTitle', 'off')
subplot(3,1,1),stem(n,u0_m,'Linewidth',2), grid
title ('Delayed Digital Unit Impulse u0(n-m)')
subplot(3,1,2),stem(n,u1_m,'Linewidth',2), grid
title ('Delayed Digital Unit Step u1(n-m)')
subplot(3,1,3),stem(n,x1_m,'Linewidth',2),xlabel('n'), grid
title ('Delayed Discrete Exponent x1(n-m)')
disp('%')
disp('%')
disp('% Для продолжения нажмите <ENTER>')
pause 
disp('%')
disp('%')
disp('% п.6. ДИСКРЕТНЫЙ ПРЯМОУГОЛЬНЫЙ ИМПУЛЬС')
disp('%')
disp('%')
disp('% Для вывода ГРАФИКОВ дискретного прямоугольного импульса нажмите <ENTER>')
pause
x3_1 = U*rectpuls(n-n0,2*n_imp); x3_1(1:n0) = 0; % ФОРМИРОВАНИЕ ИМПУЛЬСА С ПОМОЩЬЮ ФУНКЦИИ rectpuls 
x3_2 = [zeros(1,n0) U.*u1((n0+1):(n0+n_imp))...
zeros(1,N-(n0+n_imp))];     % ФОРМИРОВАНИЕ ИМПУЛЬСА С ПОМОЩЬЮ ЦИФРОВОГО ЕДИНИЧНОГО СКАЧКА
figure('Name','Discrete Rectangular and Triangular Impulses','NumberTitle', 'off')
subplot(3,1,1),stem(n,x3_1,'Linewidth',2), grid
title('Discrete Rectangular Impulse x3 1(n)')
subplot(3,1,2),stem(n,x3_2,'Linewidth',2), grid
title('Discrete Rectangular Impulse x3 2 (n)')
disp('%')
disp('%')
disp('% Для продолжения нажмите <ENTER>')
pause 
disp('%')
disp('%')
disp('% п.7. ДИСКРЕТНЫЙ ТРЕУГОЛЬНЫЙ ИМПУЛЬС')
disp('%')
disp('%')
disp('% Для вывода ГРАФИКА дискретного треугольного импульса нажмите <ENTER>')
pause
x4 = conv(x3_1,x3_1);           % ДИСКРЕТНЫЙ ТРЕУГОЛЬНЫЙ ИМПУЛЬС
L = 2*N-1;                      % ДЛИНА СВЕРТКИ
n = 0:(L-1);                    % ДИСКРЕТНОЕ НОРМИРОВАННОЕ ВРЕМЯ
subplot(3,1,3),stem(n,x4,'Linewidth',2), xlabel('n'), grid
title('Discrete Triangular Impulse x4(n)')
disp('%')
disp('%')
disp('% Для продолжения нажмите <ENTER>')
pause 
disp('%')
disp('%')
disp('% п.8. ЛИНЕЙНАЯ КОМБИНАЦИЯ ДИСКРЕТНЫХ ГАРМОНИЧЕСКИХ СИГНАЛОВ')
disp('%')
disp('%')
disp('% Для вывода ГРАФИКОВ гармонических сигналов и их линейной комбинации нажмите <ENTER>')
pause
n = 0:(5*N-1);                         % ДИСКРЕТНОЕ НОРМИРОВАННОЕ ВРЕМЯ
xi = repmat(B,length(n),1).*sin(n'*w); % МАТРИЦА ДИСКРЕТНЫХ ГАРМОНИК
ai = repmat(A,length(n),1);            % МАТРИЦА КОЭФФИЦИЕНТОВ
x5 = sum((ai.* xi)');         % ЛИНЕЙНАЯ КОМБИНАЦИЯ ДИСКРЕТНЫХ ГАРМОНИК
figure('Name','Discrete Harmonic Signals and their Linear Combination','NumberTitle', 'off')
subplot(4,1,1),stem(n, xi(:,1),'Linewidth',2), grid
title('First Discrete Harmonic Signal')
subplot(4,1,2),stem(n, xi(:,2),'Linewidth',2), grid
title('Second Discrete Harmonic Signal')
subplot(4,1,3),stem(n, xi(:,3),'Linewidth',2), grid
title('Third Discrete Harmonic Signal')
subplot(4,1,4),stem(n,x5,'Linewidth',2), xlabel('n'), grid
title('Linear Combination x5(n)') 
disp('%')
disp('%')
disp('% Для вывода СРЕДНЕГО ЗНАЧЕНИЯ, ЭНЕРГИИ и СРЕДНЕЙ МОЩНОСТИ сигнала x5 нажмите <ENTER>')
pause
mean_x5 = mean(x5);               % СРЕДНЕЕ ЗНАЧЕНИЕ СИГНАЛА
E = sum(x5.^2);                   % ЭНЕРГИЯ СИГНАЛА
P = sum(x5.^2)/length(x5);        % СРЕДНЯЯ МОЩНОСТЬ СИГНАЛА
disp('%')
disp('%')
disp(['  mean_x5 = ' num2str(mean_x5) '  E = ' num2str(E) '  P = ' num2str(P)])
disp('%')
disp('%')
disp('% Для продолжения нажмите <ENTER>')
pause 
disp('%')
disp('%')
disp('% п.9. ДИСКРЕТНЫЙ ГАРМОНИЧЕСКИЙ СИГНАЛ С ЭКСПОНЕНЦИАЛЬНОЙ ОГИБАЮЩЕЙ')
disp('%')
disp('%')
disp('% Для вывода ГРАФИКА гармонического сигнала с экспоненциальной огибающей нажмите <ENTER>')
pause 
n = 0:(N-1);                       % ДИСКРЕТНОЕ НОРМИРОВАННОЕ ВРЕМЯ
x = C.*sin(w0.*n);                 % ДИСКРЕТНЫЙ ГАРМОНИЧЕСКИЙ СИГНАЛ
x6 = x.*(abs(a).^n);               % ДИСКРЕТНЫЙ ГАРМОНИЧЕСКИЙ СИГНАЛ С ЭКСПОНЕНЦИАЛЬНОЙ ОГИБАЮЩЕЙ
figure('Name','Harmonic Signal with Exponential Envelope.  Periodic Sequence of Rectangular Impulses','NumberTitle', 'off')
subplot(2,1,1),stem(n,x6,'Linewidth',2), grid
title('Harmonic Signal with Exponential Envelope x6(n)')
disp('%')
disp('%')
disp('% Для продолжения нажмите <ENTER>')
pause 
disp('%')
disp('%')
disp('% п.10. ПЕРИОДИЧЕСКАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ ДИСКРЕТНЫХ ПРЯМОУГОЛЬНЫХ ИМПУЛЬСОВ')
disp('%')
disp('%')
disp('% Для вывода ГРАФИКА пяти периодов последовательности нажмите <ENTER>')
pause
xp = [U.*u1(1:n_imp) zeros(1,n_imp)];    % ПЕРИОД ПОСЛЕДОВАТЕЛЬНОСТИ
p = 5;                                   % ЧИСЛО ПЕРИОДОВ 
x7 =  repmat(xp,1,p);             % ПЕРИОДИЧЕСКАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ
n = 0:(length(x7)-1);             % ДИСКРЕТНОЕ НОРМИРОВАННОЕ ВРЕМЯ
subplot(2,1,2), stem(n,x7,'Linewidth',2), xlabel('n'), grid
title('Periodic Sequence of Rectangular Impulses x7(n)') 
disp('%')
disp('%')
disp('% Для продолжения нажмите <ENTER>')
pause 
disp('%')
disp('%')
disp('% п.11. РАВНОМЕРНЫЙ БЕЛЫЙ ШУМ')
disp('%')
disp('%')
disp('% Для вывода ОЦЕНОК МАТЕМАТИЧЕСКОГО ОЖИДАНИЯ и ДИСПЕРСИИ ШУМА нажмите <ENTER>')
pause
r_uniform = rand(1,10000);           % РАВНОМЕРНЫЙ БЕЛЫЙ ШУМ
mean_uniform = mean(r_uniform);      % ОЦЕНКА МАТ. ОЖИДАНИЯ ШУМА
var_uniform = var(r_uniform);        % ОЦЕНКА ДИСПЕРСИИ ШУМА
disp('%')
disp('%')
disp(['  mean_uniform = ' num2str(mean_uniform) '  var_uniform = ' num2str(var_uniform)]) 
disp('%')
disp('%')
disp('% Для вывода графика АВТОКОВАРИАЦИОННОЙ ФУНКЦИИ нажмите <ENTER>')
pause
r_r_uniform = (1/length(r_uniform)).*xcov(r_uniform);   % ОЦЕНКА АВТОКОВАРИАЦИОННОЙ ФУНКЦИИ РАВНОМЕРНОГО БЕЛОГО ШУМА
m = -(length(r_uniform)-1):(length(r_uniform)-1);       % ВЕКТОР ДИСКРЕТНЫХ СДВИГОВ ДЛЯ АВТОКОВАРИАЦИОННОЙ ФУНКЦИИ 
figure('Name','Autocovariance Function of Uniform White Noise','NumberTitle', 'off')
stem(m,r_r_uniform,'Linewidth',2), xlabel('m'), grid
title('Autocovariance Function of Uniform White Noise')
disp('%')
disp('%')
disp('% Для продолжения нажмите <ENTER>')
pause 
disp('%')
disp('%')
disp('% п.12. НОРМАЛЬНЫЙ БЕЛЫЙ ШУМ')
disp('%')
disp('%')
disp('% Для вывода ОЦЕНОК МАТЕМАТИЧЕСКОГО ОЖИДАНИЯ и ДИСПЕРСИИ шума нажмите <ENTER>')
pause
r_norm = randn(1,10000);           % НОРМАЛЬНЫЙ БЕЛЫЙ ШУМ
mean_norm = mean(r_norm);          % ОЦЕНКА МАТ. ОЖИДАНИЯ ШУМА
var_norm = var(r_norm);            % ОЦЕНКА ДИСПЕРСИИ ШУМА
disp('%')
disp('%')
disp(['  mean_norm = ' num2str(mean_norm) '  var_norm = ' num2str(var_norm)]) 
disp('%')
disp('%')
disp('% Для вывода графика АКФ нажмите <ENTER>')
pause
R_r_norm = (1/length(r_norm)).*xcorr(r_norm);   % ОЦЕНКА АКФ НОРМАЛЬНОГО БЕЛОГО ШУМА 
m = -(length(r_norm)-1):(length(r_norm)-1);     % ВЕКТОР ДИСКРЕТНЫХ СДВИГОВ ДЛЯ АКФ 
figure('Name','ACF of White Gaussian Noise','NumberTitle', 'off')
stem(m,R_r_norm,'Linewidth',2), xlabel('m'), grid
title('ACF of White Gaussian Noise')
disp('%')
disp('%')
disp('% Для продолжения нажмите <ENTER>')
pause
disp('%')
disp('%')
disp('% п.13. АДДИТИВНАЯ СМЕСЬ ДИСКРЕТНОГО ГАРМОНИЧЕСКОГО СИГНАЛА С НОРМАЛЬНЫМ БЕЛЫМ ШУМОМ')
disp('%')
disp('%')
disp('% Для вывода ГРАФИКА аддитивной смеси сигнала с шумом нажмите <ENTER>')
pause
n = 0:(N-1);                     % ДИСКРЕТНОЕ НОРМИРОВАННОЕ ВРЕМЯ
x8 = x+randn(1,N);               % АДДИТИВНАЯ СМЕСЬ СИГНАЛА С ШУМОМ
figure('Name','Mixture of Harmonic Signal and White Gaussian Noise and ACF','NumberTitle', 'off')
subplot(2,1,1),stem(n,x8,'Linewidth',2),xlabel('n'), grid
title('Mixture of Harmonic Signal and White Gaussian Noise x8(n)')
disp('%')
disp('%')
disp('% Для продолжения нажмите <ENTER>')
pause 
disp('%')
disp('%')
disp('% п.14. АКФ АДДИТИВНОЙ СМЕСИ ДИСКРЕТНОГО ГАРМОНИЧЕСКОГО СИГНАЛА С НОРМАЛЬНЫМ БЕЛЫМ ШУМОМ')
disp('%')
disp('%')
disp('% Для вывода ГРАФИКА АКФ нажмите <ENTER>')
pause 
R = (1/N).*xcorr(x8);            % ОЦЕНКА АКФ 
m = -(N-1):(N-1);                % ВЕКТОР ДИСКРЕТНЫХ СДВИГОВ ДЛЯ АКФ 
subplot(2,1,2),stem((m),R,'Linewidth',2),xlabel('m'), grid
title('ACF R(m)')
disp('%')
disp('%')
disp('% Для вывода ДИСПЕРСИИ аддитивной смеси сигнала с шумом и АКФ R(N) нажмите <ENTER>') 
pause 
disp('%')
disp('%')
disp(['  var_x8 = ' num2str(var(x8))])
disp(['  R(N) = ' num2str(R(N))])
disp('%')
disp('%')
disp('% Для продолжения нажмите <ENTER>')
pause 
disp('%')
disp('%')
disp('% п.15. НОРМАЛЬНЫЙ БЕЛЫЙ ШУМ С ЗАДАННЫМИ СТАТИСТИЧЕСКИМИ ХАРАКТЕРИСТИКАМИ')
r_normMean = randn(1,10000)+Mean;      % НОРМАЛЬНЫЙ БЕЛЫЙ ШУМ С ЗАДАННЫМ МАТЕМАТИЧЕСКИМ ОЖИДАНИЕМ
r_normVar = sqrt(Var).*randn(1,10000); % НОРМАЛЬНЫЙ БЕЛЫЙ ШУМ С ЗАДАННОЙ ДИСПЕРСИЕЙ
r_normMeanVar = sqrt(Var).*randn(1,10000)+ Mean; % НОРМАЛЬНЫЙ БЕЛЫЙ ШУМ С ЗАДАННЫМИ МАТЕМАТИЧЕСКИМ ОЖИДАНИЕМ И ДИСПЕРСИЕЙ
MAX = max([r_norm r_normMean r_normVar r_normMeanVar]); 
% МАКСИМАЛЬНОЕ ЗНАЧЕНИЕ ШУМА СРЕДИ ЧЕТЫРЕХ ЕГО РАЗНОВИДНОСТЕЙ
disp('%')
disp('%')
disp('% Для вывода ГРАФИКОВ нормального белого шума нажмите <ENTER>')
pause
figure('Name','White Gaussian Noises with different statistics','NumberTitle', 'off')
subplot(4,1,1), plot(r_norm), grid, ylim([-MAX MAX])
title(strcat([' Mean value = ',num2str(mean(r_norm)),'   Variance = ',num2str(var(r_norm))]))
subplot(4,1,2), plot(r_normMean), grid, ylim([-MAX MAX])
title(strcat([' Mean value = ',num2str(mean(r_normMean)),'   Variance = ',num2str(var(r_normMean))]))
subplot(4,1,3), plot(r_normVar), grid, ylim([-MAX MAX])
title(strcat([' Mean value = ',num2str(mean(r_normVar)),'   Variance = ',num2str(var(r_normVar))]))
subplot(4,1,4), plot(r_normMeanVar), xlabel('n'), grid, ylim([-MAX MAX])
title(strcat([' Mean value = ',num2str(mean(r_normMeanVar)),'   Variance = ',num2str(var(r_normMeanVar))]))
disp('%')
disp('%')
disp('% Для вывода ГИСТОГРАММ нормального белого шума нажмите <ENTER>')
pause
figure('Name','Histograms with different statistics','NumberTitle', 'off')
subplot(4,1,1), hist(r_norm), grid, xlim([-MAX MAX]) 
title(strcat([' Mean value = ',num2str(mean(r_norm)),'   Variance = ',num2str(var(r_norm))]))
subplot(4,1,2), hist(r_normMean), grid, xlim([-MAX MAX])
title(strcat([' Mean value =  ',num2str(mean(r_normMean)),'   Variance = ',num2str(var(r_normMean))]))
subplot(4,1,3), hist(r_normVar), grid, xlim([-MAX MAX])
title(strcat([' Mean value = ',num2str(mean(r_normVar)),'   Variance = ',num2str(var(r_normVar))]))
subplot(4,1,4),hist(r_normMeanVar), grid, xlim([-MAX MAX])
title(strcat([' Mean value = ',num2str(mean(r_normMeanVar)),'   Variance = ',num2str(var(r_normMeanVar))]))
disp('%')
disp('%')
disp('% РАБОТА ЗАВЕРШЕНА')
