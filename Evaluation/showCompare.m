clear;
clc;

% mix : the mixed signal in time domain
% source1_test: the result of speaker in time domainfrom my model (speaker by speaker, sentence by sentence)
% test1: the clean speech of speaker in time domain
% test2: the clean noise in time domain

%% Load data and settings

URL = '../Record/data_merge/SNR=5/Test1/';

load(strcat(URL, 'Original.mat'));
lstm = load(strcat(URL, 'LSTM_d=1000_sp_77.mat'));
mrnn = load(strcat(URL, 'MRNN_d=500_K=2_sp_77.mat'));

fs = 16000;
win = 0.064;
shift = 0.032;
L = 1024;

speaker = 1;
sentence= 9;

%% Generate wav file

audiowrite(strcat(URL, 'Original_mix.wav'), mix{speaker}{sentence}*30, fs);
audiowrite(strcat(URL, 'Original_clean.wav'), test1{speaker}{sentence}*30, fs);
audiowrite(strcat(URL, 'Original_noise.wav'), test2{speaker}{sentence}*30, fs);
audiowrite(strcat(URL, 'LSTM_d=1000_sp_77.wav'), lstm.source1_test{speaker}{sentence}*30, fs);
audiowrite(strcat(URL, 'MRNN_d=500_K=2_sp_77.wav'), mrnn.source1_test{speaker}{sentence}*30, fs);

%% Plot wav

figure
plot(mix{speaker}{sentence});
% axis off

figure
plot(test1{speaker}{sentence});
% axis off

figure
plot(test2{speaker}{sentence});
% axis off

figure
plot(lstm.source1_test{speaker}{sentence});
% axis off

figure
plot(mrnn.source1_test{speaker}{sentence});
% axis off

%% Plot spectrum

figure
[s, F, t] = spectrogram(test1{speaker}{sentence}, ceil(win*fs), ceil(win*fs)-ceil(shift*fs), L, 1E3, 'yaxis');
t = t*0.064;
% f = linspace(0, 1, fix(L/2)+1)*fs;
f = 0:(length(F)-1);
f = f*((fs/2)/length(F))/1000;
surf(t, f, 10*log10(abs(s)), 'edgecolor', 'none'); axis tight; view(0, 90);
colormap(jet);
set(gcf,'Units','centimeters','position',[5 5 15 5]);

figure
[s, F, t] = spectrogram(mix{speaker}{sentence}, ceil(win*fs), ceil(win*fs)-ceil(shift*fs), L, 1E3, 'yaxis');
t = t*0.064;
% f = linspace(0, 1, fix(L/2)+1)*fs;
f = 0:(length(F)-1);
f = f*((fs/2)/length(F))/1000;
surf(t, f, 10*log10(abs(s)), 'edgecolor', 'none'); axis tight; view(0, 90);
colormap(jet);
set(gcf,'Units','centimeters','position',[5 5 15 5]);

figure
[s, F, t] = spectrogram(test2{speaker}{sentence}, ceil(win*fs), ceil(win*fs)-ceil(shift*fs), L, 1E3, 'yaxis');
t = t*0.064;
% f = linspace(0, 1, fix(L/2)+1)*fs;
f = 0:(length(F)-1);
f = f*((fs/2)/length(F))/1000;
surf(t, f, 10*log10(abs(s)), 'edgecolor', 'none'); axis tight; view(0, 90);
colormap(jet);
set(gcf,'Units','centimeters','position',[5 5 15 5]);

%% Plot spectrum with label

figure
[s, F, t] = spectrogram(test1{speaker}{sentence}, ceil(win*fs), ceil(win*fs)-ceil(shift*fs), L, 1E3, 'yaxis');
t = t*0.064;
% f = linspace(0, 1, fix(L/2)+1)*fs;
f = 0:(length(F)-1);
f = f*((fs/2)/length(F))/1000;
surf(t, f, 10*log10(abs(s)), 'edgecolor', 'none'); axis tight; view(0, 90);
colormap(jet);
xlabel('Time (sec)', 'Fontsize', 15);
ylabel('Frequency (kHz)', 'Fontsize', 15);
set(gcf,'Units','centimeters','position',[5 5 15 5]);
% axis off

figure;
[s, ~, t] = spectrogram(lstm.source1_test{speaker}{sentence}, ceil(win*fs), ceil(win*fs)-ceil(shift*fs), L, 1E3, 'yaxis');
t = t*0.064;
% f = linspace(0, 1, fix(L/2)+1)*fs;
f = 0:(length(F)-1);
f = (f*((fs/2)/length(F)))/1000;
surf(t, f, 10*log10(abs(s)), 'edgecolor', 'none'); axis tight; view(0, 90);
colormap(jet);
xlabel('Time (sec)', 'Fontsize', 15);
ylabel('Frequency (kHz)', 'Fontsize', 15);
set(gcf,'Units','centimeters','position',[5 5 15 5]);
% axis off

figure;
[s, ~, t] = spectrogram(mrnn.source1_test{speaker}{sentence}, ceil(win*fs), ceil(win*fs)-ceil(shift*fs), L, 1E3, 'yaxis');
t = t*0.064;
% f = linspace(0, 1, fix(L/2)+1)*fs;
f = 0:(length(F)-1);
f = (f*((fs/2)/length(F)))/1000;
surf(t, f, 10*log10(abs(s)), 'edgecolor', 'none'); axis tight; view(0, 90);
colormap(jet);
xlabel('Time (sec)', 'Fontsize', 15);
ylabel('Frequency (kHz)', 'Fontsize', 15);
set(gcf,'Units','centimeters','position',[5 5 15 5]);
% axis off