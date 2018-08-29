clear;
clc;

%% Learning Curve
% % dnn
% % e1 = xlsread('result.xlsx', 'H2:H39');
% % e2 = xlsread('result.xlsx', 'I2:I50');
% % e3 = xlsread('result.xlsx', 'J2:J51');
% % e4 = xlsread('result.xlsx', 'K2:K41');
% % e5 = xlsread('result.xlsx', 'L2:L40');
% 
% % % lstm
% % e1 = xlsread('result.xlsx', 'N2:N42');
% % e2 = xlsread('result.xlsx', 'O2:O38');
% % e3 = xlsread('result.xlsx', 'P2:P51');
% % e4 = xlsread('result.xlsx', 'Q2:Q50');
% % e5 = xlsread('result.xlsx', 'R2:R49');
% 
% % ntm_128_20
% e1 = xlsread('result.xlsx', 'B2:B43');
% e2 = xlsread('result.xlsx', 'C2:C49');
% e3 = xlsread('result.xlsx', 'D2:D41');
% e4 = xlsread('result.xlsx', 'E2:E38');
% e5 = xlsread('result.xlsx', 'F2:F35');
% 
% figure;
% hold on;
% plot(e1,'--bs',...
%     'MarkerSize',5,...
%     'MarkerEdgeColor','b', ...
%     'MarkerFaceColor',[0.5,0.5,0.5]);
% plot(e2,'--gd',...
%     'MarkerSize',5,...
%     'MarkerEdgeColor','g', ...
%     'MarkerFaceColor',[0.5,0.5,0.5]);
% plot(e3,'--rx',...
%     'MarkerSize',5,...
%     'MarkerEdgeColor','r', ...
%     'MarkerFaceColor',[0.5,0.5,0.5]);
% plot(e4,'--kp',...
%     'MarkerSize',5,...
%     'MarkerEdgeColor','k', ...
%     'MarkerFaceColor',[0.5,0.5,0.5]);
% plot(e4,'--mh',...
%     'MarkerSize',5,...
%     'MarkerEdgeColor','m', ...
%     'MarkerFaceColor',[0.5,0.5,0.5]);
% hold off;
% xlabel('number of epoch');
% ylabel('mean square error');
% legend('train_{6}', 'train_{10}', 'train_{20}', 'train_{40}', 'train_{77}');

%% STOI

a1 = [0.6659, 0.6928, 0.718670492, 0.724186183];
b1 = [0.681790031, 0.701834276, 0.729851668144201, 0.735522607509748];
c1 = [0.707434204, 0.706197181, 0.728101492, 0.746889891515214];
d1 = [0.722727457, 0.732205019, 0.734402113, 0.759429000085512];
e1 = [0.719264808, 0.730373424241885, 0.738808331030928, 0.765274477];

a2 = [0.642, 0.7209, 0.734692664, 0.741090712222471];
b2 = [0.691883431, 0.720364132, 0.742375025, 0.749048455658782];
c2 = [0.706036087, 0.725272816, 0.745774548, 0.763538176477872];
d2 = [0.731611013, 0.755528811, 0.761201949, 0.785224265346402];
e2 = [0.749586987, 0.759711974298108, 0.769965133866458, 0.795777985];

stoi1 = [a1; b1; c1; d1; e1];
stoi2 = [a2; b2; c2; d2; e2];
stoi3 = [a1; b1; c1; d1; e1; a2; b2; c2; d2; e2];

figure;
bar(stoi1,'BaseValue', 0.62, 'ShowBaseLine', 'on');
set(gca,'xticklabel', {'6'; '10'; '20'; '40'; '77'}, 'ytickmode', 'manual', 'ytick', (0.62:0.02:0.78), 'yLim', [0.62 0.80]);
legend('DNN', 'LSTM', 'NTM', 'Recall', 'Location', 'northwest');
xlabel('Number of training speakers', 'FontSize', 15);
ylabel('STOI', 'FontSize', 15);
figure;
bar(stoi2,'BaseValue', 0.62, 'ShowBaseLine', 'on');
set(gca,'xticklabel', {'6'; '10'; '20'; '40'; '77'}, 'ytickmode', 'manual', 'ytick', (0.62:0.02:0.78), 'yLim', [0.62 0.80]);
legend('DNN', 'LSTM', 'NTM', 'Recall', 'Location', 'northwest');
xlabel('Number of training speakers', 'FontSize', 15);
ylabel('STOI', 'FontSize', 15);
% figure;
% bar(stoi3,'BaseValue', 0.62, 'ShowBaseLine', 'off');
% set(gca,'xticklabel', {'6'; '10'; '20'; '40'; '77'; '6'; '10'; '20'; '40'; '77'});
% legend('DNN', 'LSTM', 'MANN', 'Location', 'northwest');


%% Memory Size

a1 = [0.716385398, 0.714526677, 0.718670492];
b1 = [0.717741933509372, 0.715345690744655, 0.729851668144201];
c1 = [0.709656088035118, 0.6, 0.728101492];
d1 = [0.721670785605747, 0.6, 0.734402113];
e1 = [0.720765809170097, 0.6, 0.6];

a2 = [0.728320301789945, 0.731796552048787, 0.734692664];
b2 = [0.732175356936519, 0.731581850002806, 0.742375025];
c2 = [0.729608842836957, 0.6, 0.745774548];
d2 = [0.740081492033286, 0.6, 0.761201949];
e2 = [0.739920876275124, 0.6, 0.6];

%stoi1 = [a1; b1; c1; d1; e1];
%stoi2 = [a2; b2; c2; d2; e2];
stoi1 = [a1; b1];
stoi2 = [a2; b2];

figure;
bar(stoi1,'BaseValue', 0.6);
set(gca,'xticklabel', {'6'; '10'; '20'; '40'; '77'});
legend('Mem:20x128', 'Mem:20x500', 'Mem:128x128', 'Location', 'northeastoutside');
figure;
bar(stoi2,'BaseValue', 0.6);
set(gca,'xticklabel', {'6'; '10'; '20'; '40'; '77'});
legend('Mem:20x128', 'Mem:20x500', 'Mem:128x128', 'Location', 'northeastoutside');

%% Comparison of NTM

% test 1
a1 = [0.718670492107376, 0.71956900213821, 0.72121200089775];
b1 = [0.734402113248691, 0.7352, 0.6];
% test 2
a2 = [0.73469266440034, 0.737805464399891, 0.735597124057364];
b2 = [0.76120194894002, 0.759404486468508, 0.6];

stoi1 = [a1; b1];
stoi2 = [a2; b2];

figure;
bar(stoi1,'BaseValue', 0.6);
set(gca,'xticklabel', {'6'; '40'});
legend('Standord NTM', 'Multiple Attentions', 'Location Based', 'Location', 'northeastoutside');
figure;
bar(stoi2,'BaseValue', 0.6);
set(gca,'xticklabel', {'6'; '40'});
legend('Standord NTM', 'Multiple Attentions', 'Location Based', 'Location', 'northeastoutside');

%% Comparison of CNN-NTM

% test 1
a1 = [0.722819563344922, 0.718670492107376];
b1 = [0.714450580583308, 0.729851668144201];
% test 2
a2 = [0.729943096824761, 0.73469266440034];
b2 = [0.730155054067009, 0.742375025278216];

stoi1 = [a1; b1];
stoi2 = [a2; b2];

figure;
bar(stoi1,'BaseValue', 0.6);
set(gca,'xticklabel', {'6'; '10'});
legend('CNN-NTM', 'Standord NTM', 'Location', 'northeastoutside');
figure;
bar(stoi2,'BaseValue', 0.6);
set(gca,'xticklabel', {'6'; '10'});
legend('CNN-NTM', 'Standord NTM', 'Location', 'northeastoutside');


