

clear; clc; close all;

%% ---------------- Parameters ----------------
N       = 100;
mu      = 1.5;

eps1 = 0.47;  eps2 = 0.475;  eps3 = 0.48;
eps4 = 0.503;                % Prediction target

dt_save = 0.02;
k       = 2;
d       = 2;

M_train = 3000;
m_pred  = 10000;

reg = 9e-3;

%% ---------------- Data generation ----------------
fprintf('Generating coupled van der Pol ring data...\n');

n_need  = M_train + m_pred + k + 10;
t_total = n_need * dt_save;
tspan   = 0:dt_save:t_total;

rng(42);
x0 = randn(N,1);
y0 = randn(N,1);
u0 = [x0; y0];

opts = odeset('RelTol',1e-8,'AbsTol',1e-10);
[~, X1] = ode45(@(tt,u) ring_vdp_rhs(u,N,mu,eps1), tspan, u0, opts);
[~, X2] = ode45(@(tt,u) ring_vdp_rhs(u,N,mu,eps2), tspan, u0, opts);
[~, X3] = ode45(@(tt,u) ring_vdp_rhs(u,N,mu,eps3), tspan, u0, opts);

t = tspan';
fprintf('  Number of samples: %d, total duration: %.2f\n', length(t), t(end));

%% ---------------- Feature dimensions ----------------
L  = 6*k + 1;                 
n2 = L * (L + 1) / 2;
n3 = L * (L + 1) * (L + 2) / 6;
n_feat = 1 + L + n2 + n3;

fprintf('L=%d, quadratic terms=%d, cubic terms=%d, total features=%d\n', L, n2, n3, n_feat);

s_train = k : (M_train + k - 1);
n_train = length(s_train);

%% ---------------- Training ----------------
fprintf('Constructing features and training...\n');

G = zeros(n_feat, n_feat);
C = zeros(d, n_feat);

X_all  = {X1, X2, X3};
eps_all = {eps1, eps2, eps3};

for q = 1:numel(X_all)
    Xq   = X_all{q};
    epsq = eps_all{q};
    for i = 1:N
        F_i = zeros(n_feat, n_train);
        Y_i = zeros(d, n_train);
        for j = 1:n_train
            s = s_train(j);
            z = get_local_linear(Xq, i, s, k, N, epsq);
            F_i(:, j) = build_feature(z, L, n2, n3);
            Y_i(:, j) = [Xq(s+1,i)   - Xq(s,i);
                         Xq(s+1,N+i) - Xq(s,N+i)];
        end
        G = G + F_i * F_i';
        C = C + Y_i * F_i';
    end
end

W = C / (G + reg * eye(n_feat));
fprintf('  Shared W trained, size %dx%d\n', size(W,1), size(W,2));

%% ---------------- Closed-loop prediction ----------------
fprintf('Closed-loop prediction for %d steps (eps=%.2f)...\n', m_pred, eps4);

X_hist = zeros(2*N, k);
for lag = 1:k
    X_hist(:, lag) = u0; 
end

X_pred = zeros(2*N, m_pred);

for v = 1:m_pred
    x_cur = X_hist(:, 1);
    x_new = zeros(2*N, 1);

    for i = 1:N

        z = zeros(6*k, 1);
        im1 = mod(i-2, N) + 1;
        ip1 = mod(i,   N) + 1;

        for lag = 1:k
            base = 6*(lag-1);
            z(base+1) = X_hist(im1,   lag);
            z(base+2) = X_hist(N+im1, lag);
            z(base+3) = X_hist(i,     lag);
            z(base+4) = X_hist(N+i,   lag);
            z(base+5) = X_hist(ip1,   lag);
            z(base+6) = X_hist(N+ip1, lag);
        end

        z = [eps4; z];        
        f = build_feature(z, L, n2, n3);
        dx = W * f;

        x_new(i)   = x_cur(i)   + dx(1);
        x_new(N+i) = x_cur(N+i) + dx(2);
    end

    X_pred(:, v) = x_new;

    if any(abs(x_new) > 1e4)
        fprintf('  Prediction diverged at step %d\n', v);
        X_pred = X_pred(:,1:v);
        break;
    end

    X_hist = [x_new, X_hist(:, 1:end-1)];
end


%% ---------------- Visualization ----------------
figure(1);
plot(X_pred(67,1:3894), X_pred(N+67,1:3894),'r', 'LineWidth',1);   
xlabel('x'); ylabel('y');
xlim([-4 3]); ylim([-6 12]);
grid on;



%% ======================= Subfunctions =======================
function du = ring_vdp_rhs(u, N, mu, eps_val)
    x = u(1:N);
    y = u(N+1:2*N);

    f = y;
    g = -mu*(x.^2 - 1).*y - x;

    im1 = mod((1:N)-2, N) + 1;
    ip1 = mod((1:N),   N) + 1;

    dx = f + (eps_val/2)*(f(im1) + f(ip1) - 2*f);
    dy = g + (eps_val/2)*(g(im1) + g(ip1) - 2*g);

    du = [dx; dy];
end

function z = get_local_linear(X, i, s, k, N, eps_val)
    im1 = mod(i-2, N) + 1;
    ip1 = mod(i,   N) + 1;

    z = zeros(6*k, 1);
    for lag = 0:k-1
        ss   = s - lag;
        base = 6*lag;
        z(base+1) = X(ss, im1);
        z(base+2) = X(ss, N+im1);
        z(base+3) = X(ss, i);
        z(base+4) = X(ss, N+i);
        z(base+5) = X(ss, ip1);
        z(base+6) = X(ss, N+ip1);
    end
    z = [eps_val; z];      
end

function f = build_feature(z, L, n2, n3)
    f = zeros(1 + L + n2 + n3, 1);
    f(1) = 1;
    f(2:L+1) = z;

    idx = L + 1;
    for a = 1:L
        for b = a:L
            idx = idx + 1;
            f(idx) = z(a) * z(b);
        end
    end
    for a = 1:L
        for b = a:L
            for c = b:L
                idx = idx + 1;
                f(idx) = z(a) * z(b) * z(c);
            end
        end
    end
end