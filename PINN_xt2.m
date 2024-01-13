load("Waves_lintrans_small.mat");
%% Resample
dt=1;
dx=2;
rng(0);
Xn=Waves;
noise = sqrt(var(Xn(:))*0.01)*randn(size(Xn)); 
Xn = Xn+noise;
x_lb = 3;
t_lb = 5;

x_onecol = x_lb: dx: x_lb+20*dx; % first change x
t_onerow = t_lb: dt: t_lb+190*dt;
t_of = t_lb: 0.5*dt: t_lb+190*dt;
X0 = repmat(x_onecol,1,length(t_onerow));
x_onecolf = x_lb: 0.5*dx: x_lb+20*dx; 
X0f = repmat(x_onecolf,1,length(t_of));
figure
plot(X0f)
title('X0f')
T0f=[];
for i=1:length(t_of)
    T0f = [T0f, t_of(i)*ones(1,length(x_onecolf))];
end
T0=[];
for i=1:length(t_onerow)
    T0 = [T0, t_onerow(i)*ones(1,length(x_onecol))];
end
figure
plot(T0f)
title('T0f')
Usel = Xn(3:23, 5:1:195);
figure
imagesc(Xn)
hold on
plot([4.5,4.5],[2.5,23.5],'r')
hold on
plot([195.5,195.5],[2.5,23.5],'r')
hold on
plot([4.5,195.5],[2.5,2.5],'r')
hold on
plot([4.5,195.5],[23.5,23.5],'r')
ylabel('x (\rm dx)','FontSize',14,'Interpreter','latex')
xlabel('t (\rm dt)','FontSize',14,'Interpreter','latex')
ax=gca;
ax.FontSize = 14;
ax.TickLabelInterpreter='latex';
colorbar
caxis([-2 2])
exportgraphics(ax,'noisy-measurements1.png')

U0 = reshape(Usel, 1, []);
% Ground truth, only used for plotting 
GT = -ones(21,1);
GT(1:3) = -2.25;
GT(4:8) = [-1.96,-1.69,-1.44,-1.21,-1];
GT(14:18) = [-1.44,-1.96,-2.56,-3.24,-4];
GT(19:end) = -4;
%% 
ds = arrayDatastore([X0f' T0f']);
%%
numLayers = 9;
numNeurons = 20;

parameters = struct;

sz = [numNeurons 2];
parameters.fc1.Weights = initializeHe(sz,2);
parameters.fc1.Bias = initializeZeros([numNeurons 1]);

parameters.lambda = initializeZeros([1, length(x_onecolf)]); %%

for layerNumber=2:numLayers-1
    name = "fc"+layerNumber;

    sz = [numNeurons numNeurons];
    numIn = numNeurons;
    parameters.(name).Weights = initializeHe(sz,numIn);
    parameters.(name).Bias = initializeZeros([numNeurons 1]);
end

sz = [1 numNeurons];
numIn = numNeurons;
parameters.("fc" + numLayers).Weights = initializeHe(sz,numIn);
parameters.("fc" + numLayers).Bias = initializeZeros([1 1]);
%%
numEpochs = 400000;
miniBatchSize = length(X0f);

executionEnvironment = "auto";

initialLearnRate = 0.01;
decayRate = 0.0000;
%%
mbq = minibatchqueue(ds, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFormat="BC", ...
    OutputEnvironment=executionEnvironment);
%%
X0 = dlarray(X0,"CB");
T0 = dlarray(T0,"CB");
U0 = dlarray(U0);

averageGrad = [];
averageSqGrad = [];

accfun = dlaccelerate(@modelLoss);
%%
iteration = 0;
LAMBDA = zeros(numEpochs,length(x_onecolf));
LOSS = zeros(numEpochs,1);
for epoch = 1:numEpochs
    tic
    LAMBDA(epoch,:) = extractdata(parameters.lambda);
    reset(mbq);
    epoch
    while hasdata(mbq)
        iteration = iteration + 1;

        XT = next(mbq);
        X = XT(1,:);
        T = XT(2,:);

        [loss,gradients, Ux] = dlfeval(accfun,parameters,X,T,X0,T0,U0);
        
        % Update learning rate.
        learningRate = initialLearnRate / (1+decayRate*iteration);

        % Update the network parameters using the adamupdate function.
        [parameters,averageGrad,averageSqGrad] = adamupdate(parameters,gradients,averageGrad, ...
            averageSqGrad,iteration,learningRate);
    end
    % Plot training progress.
    loss = double(gather(extractdata(loss)));
    LOSS(epoch) = loss;
    if(epoch==5 || mod(epoch,1000)==0)
        fname = strcat('Saved1D/',strcat(num2str(epoch),'_test1D_2.mat'));
        save(fname);
    end
    toc
end
%% Plot
figure
plot(-LAMBDA2(:,1:2:end))
xlabel('Epoch','Interpreter','latex')
ylabel('$\widehat{\lambda}_{m1}$','Interpreter','latex')
ax=gca
ax.FontSize=14
ax.TickLabelInterpreter='latex'
ylim([-0.5 4.5])
%exportgraphics(ax,'lambda-epoch1.png')

figure
plot(3:23,-GT)
hold on
plot(3:23,-extractdata(parameters.lambda2(1:2:end)))
xlim([3,23])
legend({'True','Recovered by LSQ','Recovered by LSQ with TVR','Recovered by SD-PINN'},'Interpreter','latex')
xlabel('x (\rm dx)','Interpreter','latex')
ylabel('$c^2$','Interpreter','latex')
xticks([3,5,10,15,20,23])
ax=gca
ax.FontSize=14
ax.TickLabelInterpreter='latex'
%exportgraphics(ax,'recovered-paras-noisymeas1.png')
