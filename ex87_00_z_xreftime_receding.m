% Solves the following minimization problem using direct single shooting
%  minimize       1/2*integral{t=0 until t=T}(x1^2 + x2^2 + u^2) dt
%  subject to     dot(x1) = (1-x2^2)*x1 - x2 + u,   x1(0)=0, x1(T)=0
%                 dot(x2) = x1,                     x2(0)=1, x2(T)=0
%  with T=10
%
% This example can be found in CasADi's example collection in MATLAB/Octave,
% Python and C++ formats.
%
% Joel Andersson, UW Madison 2017
%
% States and control

clear
clc
close all

% addpath(genpath('C:\Users\helon\Documents\MATLAB\casadi'));
import casadi.*

x1 = SX.sym('x1');
x2 = SX.sym('x2');
u  = SX.sym('u');

% Model equations
x1_dot = (1-x2^2)*x1 - x2 + u;
x2_dot = x1;

% Define the problem structure
ocp = struct('x',[x1; x2],'u',u, 'ode', [x1_dot; x2_dot]);

% Problem dimensions
N = 10; % MPC window size
nx = numel(ocp.x);
nu = numel(ocp.u);

% Specify problem data
data = struct(...
    'T', 20,...                 % simulation time
    'Ts', 0.05,...               % simulation time
    'x0', [0; 1],...            % initial conditions
    'x2ref', [],...             % reference for state x(2)
    'u_min', -1,...             % input bounds
    'u_max', 1,...
    'x_min', [-0.21;-inf],...   % state bounds
    'x_max', [inf;inf],...
    'u_guess', 0, ...           % initial guess
    'x_guess', [0;0],...
    'tol',1e-8);                % tolerance for solver

% time vector
data.tvec = 0:data.Ts:(data.T+N*data.Ts); % we added here also the instants in the future of the last simulation time T
% create vector for reference on time
data.x2ref = 0*data.tvec;
data.x2ref(data.tvec<10) = 0.1;
data.x2ref(data.tvec>=10) = -0.1;
% plot(data.tvec,data.x2ref);

% Problem independent code from here on


% CVODES from the SUNDIALS suite
dae = struct('x',ocp.x,'p',ocp.u,'ode',ocp.ode);
optsInt = struct('tf',data.Ts,'reltol',data.tol,'abstol',data.tol,'verbose',false);
F = integrator('F', 'cvodes', dae, optsInt);

% Start with an empty NLP
w   = {}; % Variables
lbw = {}; % Lower bound on w
ubw = {}; % Upper bound on w
ubg = {}; % Upper bound on g
lbg = {}; % Lower bound on g
w0  = {}; % Initial guess for w
g   = {}; % Equality constraints
J   = 0;  % Objective function
P   = {}; % input constants for calculating the cost
% --- explanation for P
% parameters (which include the initial condition at the beggining of each
% MH window and the reference state of the robot)
% P contains the constants that are used each run of the optimization 
% algorithm, used to calculate the cost function
% --- explanation for P

% Expressions corresponding to the trajectories we want to plot
x_plot = {};
u_plot = {};

% Initial conditions
xk = MX.sym('x0', nx);
w{end+1} = xk;
lbw{end+1} = data.x_min;
ubw{end+1} = data.x_max;
w0{end+1} = data.x_guess;

Pk = MX.sym('P0',nx);
g{end+1} = xk - Pk; % enforces initial condition
lbg{end+1} = zeros(nx,1);
ubg{end+1} = zeros(nx,1);
P{end+1} = Pk;

% variable for ploting
x_plot{end+1} = xk;

% Loop over all times
for k=1:N
    % Declare local control
    uk = MX.sym(['u' num2str(k)], nu);
    w{end+1} = uk;
    lbw{end+1} = data.u_min;
    ubw{end+1} = data.u_max;
    w0{end+1} = data.u_guess;
    
    % Simulate the system forward in time
    Fk = F('x0', xk, 'p', uk);
    xnext = Fk.xf;
    
    % Add contribution to objective function
    Pk = MX.sym(['P' num2str(k)],1); % ref. on x2
    % J = J + xnext(1)^2 + xnext(2)^2 + uk^2; % regulation objective function
    % J = J + (xnext(2) - P(3))^2 + uk^2; % regulation objective function
    J = J + 1e3*(xnext(2) - Pk)^2 + uk^2; % regulation objective function
    P{end+1} = Pk;
    
    % Enforce state bounds
    g{end+1} = xnext(1);
    lbg{end+1} = -0.21;
    ubg{end+1} = 1;
    
    % Declare local state
    xk = MX.sym(['x' num2str(k)], nx);
    w{end+1} = xk;
    lbw{end+1} = data.x_min;
    ubw{end+1} = data.x_max;
    w0{end+1} = data.x_guess;
    g{end+1} = xk - xnext;
    lbg{end+1} = zeros(nx,1);
    ubg{end+1} = zeros(nx,1);
    
    % variable for ploting
    u_plot{end+1} = uk;
    x_plot{end+1} = xk;
end

% Enforce terminal conditions
% g{end+1} = xk - data.xN;
% g{end+1} = xk(2) - P(3);
% lbg{end+1} = 0;
% ubg{end+1} = 0;

% Concatenate variables and constraints
w = vertcat(w{:});
lbw = vertcat(lbw{:});
ubw = vertcat(ubw{:});
lbg = vertcat(lbg{:});
ubg = vertcat(ubg{:});
w0 = vertcat(w0{:});
g = vertcat(g{:});
P = vertcat(P{:});

% Formulate the NLP solver object
nlp = struct('x', w, 'g', g, 'f', J,'p',P);
opts = struct;
% opts.ipopt.max_iter = 100;
opts.ipopt.print_level = 0;%0,3
opts.print_time = 0;
% opts.ipopt.acceptable_tol =1e-4;
% opts.ipopt.acceptable_obj_change_tol = 1e-4;

solver = nlpsol('solver', 'ipopt', nlp, opts);

% Create a function that maps the NLP decision variable (w) to the trajectories (x and u)
traj = Function('traj', {w}, {horzcat(x_plot{:}), horzcat(u_plot{:})},{'w'}, {'x', 'u'});
% Create a function that maps the x and u to w (for shifting warm start)
warm = Function('warm', {horzcat(x_plot{:}), horzcat(u_plot{:})}, {w},{'x', 'u'},{'w'});

xsim = data.x0;
usim = [];
xref = [];
tvec = [];
dtvec = [];
for k = 0 : 1 : (data.T/data.Ts)
% for t = 0:data.Ts:data.T % MPC loop
    
    % printouts
    t = k*data.Ts;
    pct = t/data.T*100;
    if ~rem(pct,10)
        fprintf('Time %2.1f/%d \t %d %%\n',t,data.T,pct);
    end
    
    % select reference signal within prediction window
    x2refvec = data.x2ref(k+1: k+N);
    
    % Solve the NLP and extract solution
    tic
    sol = solver('x0', w0, 'lbx', lbw, 'ubx', ubw, 'lbg', lbg, 'ubg', ubg,'p',[xsim(:,end);x2refvec(:)]);
    dt = toc;
    [x_opt, u_opt] = traj(sol.x);
    x_opt = full(x_opt);
    u_opt = full(u_opt);
    
    % Simulate the system forward in time
    uk = u_opt(:,1);                    % select 1st control signal
    Fk = F('x0', xsim(:,end), 'p', uk); % simulate in open loop
    xk1 = full(Fk.xf);                  % extract simulation
    
    % save open loop simulation for ploting
    xsim = [xsim xk1];
    usim = [usim uk];
    tvec = [tvec t];
    dtvec = [dtvec dt];
    
    % warm start next optimization
    w0 = warm([x_opt(:,2:end) x_opt(:,end)],[u_opt(:,2:end) u_opt(:,end)]); % the last element is just a repetition
end
% discard last simulation (it's about k+1)
xsim = xsim(:,1:end-1);

% Plot the solution
figure();
clf;
hold on;
grid on;
tgrid = linspace(0, data.T, N);
plot(tvec, xsim(1,:), 'r-','linewidth',2);
plot(tvec, xsim(2,:), 'b-','linewidth',2);
stairs(data.tvec(1:end-N), data.x2ref(1:end-N), 'k--');
stairs(tvec, usim(1,:));
xlabel('t')
legend('x1','x2','x2ref','u')

figure
plot(tvec,dtvec)
