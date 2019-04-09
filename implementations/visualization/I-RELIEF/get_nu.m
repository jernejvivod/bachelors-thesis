function [nu] = get_nu(gamma_vals, mean_m_vals, mean_h_vals, nrow)
    % function [nu] = get_nu(gamma_vals, mean_m_vals, mean_h_vals, nrow)
    %
    % get nu vector (See article, pg. 4).
    %
    % gamma_vals --- gamma values of training examples
    % mean_m_vals --- mean m values of trianing examples
    % mean_h_vals --- mean h values of training examples
    % nrow --- number of training examples
    
    nu = (1/nrow) * sum(gamma_vals .* (mean_m_vals - mean_h_vals), 1);
end