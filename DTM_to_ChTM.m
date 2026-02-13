function [ChTM_Macro, ChTM_Micro] = DTM_to_ChTM(DTM_In, order, torso_th, mD_th)
% DTM_TO_CHTM Generates Chebyshev-Time Maps (ChTM) from Radar DTM.
%
% Inputs:
%   DTM_In - Real-valued DTM magnitude matrix.
%   order  - Order of Chebyshev polynomials.
%   torso_th - Threshold for strong torso detection.
%   mD_th - Threshold for micro-Doppler limb detection.
%
% Outputs:
%   ChTM_Macro - (order + 1) x Time matrix for Torso features.
%   ChTM_Micro - (order + 1) x Time matrix for Global/Micro-Doppler signature.

    % Initialization
    img_norm = mat2gray(DTM_In); % Normalize input to [0, 1] for consistent thresholding
    [rows, cols] = size(img_norm); 

    % Envelope Extraction
    img_smooth = imgaussfilt(img_norm, 2.0); % Apply Gaussian smoothing to mitigate speckle noise
    max_val = max(img_smooth(:));
    thresh_torso = torso_th * max_val;
    thresh_mD = mD_th * max_val; % Thresholding
    
    % Allocate raw indices
    raw_mD_up = nan(1, cols); raw_mD_low = nan(1, cols);
    raw_torso_up = nan(1, cols); raw_torso_low = nan(1, cols);
    
    % Column-wise boundary detection
    for c = 1:cols
        prof = img_smooth(:, c);
        
        % Micro Envelopes
        idx_mD = find(prof > thresh_mD);
        if ~isempty(idx_mD)
            raw_mD_up(c) = idx_mD(1);
            raw_mD_low(c) = idx_mD(end);
        end
        
        % Macro Envelopes
        idx_torso = find(prof > thresh_torso);
        if ~isempty(idx_torso)
            raw_torso_up(c) = idx_torso(1);
            raw_torso_low(c) = idx_torso(end);
        end
    end
    
    % Post-processing: Fill gaps, smooth, and remove outliers
    uppermDEnvelope = smooth_and_fill(raw_mD_up, cols);
    lowermDEnvelope = smooth_and_fill(raw_mD_low, cols);
    uppertorsoEnvelope = smooth_and_fill(raw_torso_up, cols);
    lowertorsoEnvelope = smooth_and_fill(raw_torso_low, cols);

    % ChTM Generation
    ChTM_Macro = zeros(order + 1, cols);
    ChTM_Micro = zeros(order + 1, cols);
    
    for c = 1:cols
        % Macro ChTM
        r_start = max(1, floor(uppertorsoEnvelope(c)));
        r_end   = min(rows, ceil(lowertorsoEnvelope(c)));
        
        if r_end > r_start
            segment = img_norm(r_start:r_end, c);
            ChTM_Macro(:, c) = compute_chebyshev_coeffs(segment, order);
        end
        
        % Micro ChTM
        r_start = max(1, floor(uppermDEnvelope(c)));
        r_end   = min(rows, ceil(lowermDEnvelope(c)));
        
        if r_end > r_start
            segment = img_norm(r_start:r_end, c);
            ChTM_Micro(:, c) = compute_chebyshev_coeffs(segment, order);
        end
    end
end

function y_out = smooth_and_fill(y_in, n_cols)
    % Handles NaNs, fills gaps, removes outliers, and smooths curves
    y_filled = y_in;
    
    % Fallback to centerline if detection failed completely
    if all(isnan(y_filled))
        y_filled = ones(size(y_filled)) * (n_cols/2);
    else
        % Fill edges
        if isnan(y_filled(1)), y_filled(1) = nanmean(y_filled); end
        if isnan(y_filled(end)), y_filled(end) = nanmean(y_filled); end
        % Linear interpolation for internal gaps
        y_filled = fillmissing(y_filled, 'linear');
    end
    
    % Outlier removal and smoothing
    y_med = movmedian(y_filled, 5);
    y_out = smoothdata(y_med, 'rloess', 15);
end

function coeffs = compute_chebyshev_coeffs(signal, order)
    % Orthogonal projection of signal onto Chebyshev polynomials
    N = length(signal);
    if N < 2
        coeffs = zeros(order+1, 1);
        return;
    end
    
    % Map indices to Chebyshev domain [-1, 1]
    x = linspace(-1, 1, N)';
    
    % Generate Basis Matrix T using Recurrence
    T = zeros(N, order + 1);
    T(:, 1) = 1; % T0
    T(:, 2) = x; % T1
    for n = 3:(order + 1)
        T(:, n) = 2 * x .* T(:, n-1) - T(:, n-2);
    end
    
    % Least Squares Projection
    raw_coeffs = (double(signal)' * T)';
    coeffs = raw_coeffs / N;
end