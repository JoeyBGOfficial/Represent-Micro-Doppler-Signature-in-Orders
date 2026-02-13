%% Synthetic Signal Demo Script for Verifying DTM_to_ChTM Function
% Original Author: JoeyBG.
% Merged & Improved By: JoeyBG.
% Date: 2026-02-07.
% Affiliation: Beijing Institute of Technology.
% Platform: MATLAB R2025b.
% 
% Description:
%   Generates synthetic Micro-Doppler data and verifies the functionality
%   of the DTM_to_ChTM extraction function.
%   Visualizes results using the specific style from SimH_ChTM_Processing.m.

%% Initialization
clear all; 
close all; 
clc;
disp('---------- © Author: JoeyBG © ----------');

%% 1. Visualization Parameters
Font_Name = 'Palatino Linotype'; 
Font_Size_Basis = 12;
Font_Size_Axis = 13;
Font_Size_Title = 14;
Font_Weight_Title = 'bold';

% Custom Colormap from SimH_ChTM_Processing.m
JoeyBG_Colormap = [0.6196 0.0039 0.2588; 0.8353 0.2431 0.3098; 0.9569 0.4275 0.2627; ...
                   0.9922 0.6824 0.3804; 0.9961 0.8784 0.5451; 1.0000 1.0000 0.7490; ...
                   0.9020 0.9608 0.5961; 0.6706 0.8667 0.6431; 0.4000 0.7608 0.6471; ...
                   0.1961 0.5333 0.7412; 0.3686 0.3098 0.6353];             % My favorite colormap
JoeyBG_Colormap_Flip = flip(JoeyBG_Colormap);

% Chebyshev order definition
Chebyshev_Order = 64;
torso_th = 0.5;
mD_th = 0.1;

%% 2. Generate Synthetic DTM Data
fprintf('Generating Synthetic Micro-Doppler Data...\n');

% Time-Frequency Grid
cols = 200; % Time samples
rows = 200; % Doppler bins
t = linspace(0, 1, cols);
f = linspace(-100, 100, rows);
[T_grid, F_grid] = meshgrid(t, f);

% Torso Signal
f_torso = 5 * sin(2*pi*1*T_grid); 
Sig_Torso = exp( -((F_grid - f_torso).^2) / (2 * 5^2) ); % Gaussian width 5

% Limb Signal
f_limb = 40 * sin(2*pi*1*T_grid);
Sig_Limb = 0.4 * exp( -((F_grid - f_limb).^2) / (2 * 3^2) ); % Gaussian width 3

% Combine + Noise
DTM_Synthetic = Sig_Torso + Sig_Limb;
% Add random noise
Noise = 0.05 * rand(rows, cols);
DTM_Synthetic = DTM_Synthetic + Noise;

% Normalize to [0, 1] as expected by the function
DTM_Input = mat2gray(DTM_Synthetic);

%% 3. Execute DTM_to_ChTM Function
fprintf('Running DTM_to_ChTM Extraction...\n');

% Calling the DTM_to_ChTM function for evaluation
[ChTM_Macro, ChTM_Micro] = DTM_to_ChTM(DTM_Input, Chebyshev_Order,torso_th,mD_th);

%% 4. Post-Processing for Visualization
% As per original script logic for visualization
norm_01 = @(x) (x - min(x(:))) / (max(x(:)) - min(x(:)));

% Log transform to reveal details
ChTM_Macro_Log = log10(abs(ChTM_Macro) + 1e-6);
ChTM_Micro_Log = log10(abs(ChTM_Micro) + 1e-6);

% Normalize for display
ChTM_Macro_Vis = norm_01(ChTM_Macro_Log);
ChTM_Micro_Vis = norm_01(ChTM_Micro_Log);

%% 5. Visualization
figure('Name', 'Demo: DTM to ChTM Verification', 'Color', 'w', 'Position', [100, 100, 1400, 400]);
cheb_axis = 0:Chebyshev_Order;

% Subplot 1: Input DTM
ax1 = subplot(1, 3, 1);
imagesc(t, f, DTM_Input);
axis xy; colormap(ax1, JoeyBG_Colormap_Flip);
title('Synthetic Input DTM', 'FontName', Font_Name, 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis);
ylabel('Doppler (Hz)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis);
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'LineWidth', 1.5, 'Box', 'on');

% Subplot 2: ChTM (Macro)
ax2 = subplot(1, 3, 2);
imagesc(t, cheb_axis, ChTM_Macro_Vis);
axis xy; colormap(ax2, JoeyBG_Colormap_Flip);
clim([0.6, 1.0]); % Clip limit from original script
title('ChTM (Macro: Torso)', 'FontName', Font_Name, 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis);
ylabel('Chebyshev Order (n)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis);
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'LineWidth', 1.5, 'Box', 'on');

% Subplot 3: ChTM (Micro)
ax3 = subplot(1, 3, 3);
imagesc(t, cheb_axis, ChTM_Micro_Vis);
axis xy; colormap(ax3, JoeyBG_Colormap_Flip);
clim([0.6, 1.0]); % Clip limit from original script
title('ChTM (Micro: Global)', 'FontName', Font_Name, 'FontSize', Font_Size_Title, 'FontWeight', Font_Weight_Title);
xlabel('Time (s)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis);
ylabel('Chebyshev Order (n)', 'FontName', Font_Name, 'FontSize', Font_Size_Axis);
set(gca, 'FontName', Font_Name, 'FontSize', Font_Size_Basis, 'LineWidth', 1.5, 'Box', 'on');

fprintf('Verification Complete. Figure generated.\n');