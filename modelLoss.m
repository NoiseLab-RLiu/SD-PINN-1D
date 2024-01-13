function [loss,gradients, Ux] = modelLoss(parameters,X,T,X0,T0,U0)
% Make predictions with the initial conditions.
U = model(parameters,X,T);

% Calculate derivatives with respect to X and T.
gradientsU = dlgradient(sum(U,"all"),{X,T},EnableHigherDerivatives=true);
Ux = gradientsU{1};
Ut = gradientsU{2};

% Calculate second-order derivatives with respect to X.
Uxx = dlgradient(sum(Ux,"all"),X,EnableHigherDerivatives=true);
Utt = dlgradient(sum(Ut,"all"),T,EnableHigherDerivatives=true);

% Calculate lossF. Enforce Burger's equation.
f = Utt + repmat(parameters.lambda,1,(190*2+1)).*Uxx; 

zeroTarget = zeros(size(f), "like", f);
lossF = 10*mse(f, zeroTarget);
% the smoothness loss is incorporated into lossF
for i=1:2:length(parameters.lambda)-2
    lossF = lossF+10*mse(parameters.lambda(i)+parameters.lambda(i+2)-2*parameters.lambda(i+1), 0, 'DataFormat','U');
end

loss_s = 0; % loss for the sign
for i=1:length(parameters.lambda)
    loss_s = loss_s + 1e1*relu(parameters.lambda2(i));
end

loss_b = 10*(mse(parameters.lambda(1), -2.25, 'DataFormat','U') + mse(parmeters.lambda(end), -4, 'DataFormat','U'));

% Calculate lossU. Enforce initial and boundary conditions.
U0Pred = model(parameters,X0,T0);
lossU = mse(U0Pred, U0);

% Combine losses.
loss = lossF + lossU + loss_s + loss_b;

% Calculate gradients with respect to the learnable parameters.
gradients = dlgradient(loss,parameters);

end
