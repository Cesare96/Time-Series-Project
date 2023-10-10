clear 
close all
clc
warning off

%% data management 

table = readtable("ITACPIALLMINMEI.csv",'Delimiter','comma');
cpi = table2array(table(:,2));
TF = isoutlier(cpi,'mean'); 
outlier = find(TF,248);

%% Calculation inflation rate 

T = length(cpi);
time = table2array(table(:,1)); 
figure
subplot(2,1,1)
plot(time,cpi)
title('Consumer Price Index of Italy')
xlabel('Years')

rate = nan(1,T-1);
for t = 2:T
    rate(t) = (cpi(t)-cpi(t-1))/cpi(t-1);
end
rate(:,1) = [];

subplot(2,1,2)
plot(time(1:end-1),rate)
title('Infaltion Rate of Italy')
xlabel('Years')
ylabel('Percentage')

%% ACF and PACF

figure
subplot(2,1,1)
autocorr(rate)
subplot(2,1,2)
parcorr(rate)
ylim([-0.3,1])

[h0,pValue0] = lbqtest(rate);

%% Partion of the sample

tstar = 100; 
numObs = length(rate);
rate1 = rate(1:tstar);
rate2 = rate(tstar+1:numObs);


%% Estimation best ARMA models

pMax = 4;
qMax = 4;
LogL = nan(pMax,qMax);
PQ = nan(pMax,qMax);

for p = 1:pMax
    for q = 1:qMax
        mdl = arima(p,0,q);
        [estMdl,~,LogL(p,q)] = estimate(mdl,rate1','Display','off');
        PQ(p,q) = p+q;
    end
end

logL = LogL(:);
pq = PQ(:);
[~,bic] = aicbic(logL,pq+2, length(rate1));
BIC = reshape(bic,pMax,qMax);

minBIC = min(BIC,[],'all');

[minpBIC,minqBIC] = find(minBIC == BIC);

%% Estimation the residuals of the best models (Diagnostics)

% % ARMA(1,1)

mdl1 = arima(1,0,1);
estMdl1 = estimate(mdl1,rate1','Display','off');

res1 = nan(1,tstar);
chat1 = estMdl1.Constant;
phihat = cell2mat(estMdl1.AR);
thetahat = cell2mat(estMdl1.MA);
res1(1) = 0;

for t = 2:tstar 
    res1(t) = rate1(t) - chat1 - phihat*rate1(t-1) - thetahat*res1(t-1);
end

% % ARMA(2,2)

mdl2 = arima(2,0,2);
estMdl2 = estimate(mdl2,rate1','Display','off');

res2 = nan(1,length(rate1));
chat2 = estMdl2.Constant;
phi1hat = cell2mat(estMdl2.AR(1,1));
phi2hat = cell2mat(estMdl2.AR(1,2));
theta1hat = cell2mat(estMdl2.MA(1,1));
theta2hat = cell2mat(estMdl2.MA(1,2));
res2(1) = 0;
res2(2) = 0;
 for t = 3:length(rate1)
     res2(t) = rate1(t) - chat2 - phi1hat*rate1(t-1) - phi2hat*rate1(t-2) - theta1hat*res2(t-1) - theta2hat*res2(t-2);
 end

 figure
 subplot(2,1,1)
 plot(res1)
 title('Residulas ARMA(1,1)')
 subplot(2,1,2)
 plot(res2)
 title('Residuals ARMA(2,2)')

figure
subplot(2,1,1)
autocorr(res1)
title('Correlogram residuals ARMA(1,1)')
subplot(2,1,2)
autocorr(res2)
title('Correlogram residuals ARMA(2,2)')

[h1,pValue1] = lbqtest(res1);
[h2,pValue2] = lbqtest(res2);

%% In-sample forecast ARMA(1,1) (h=1, h=2, h=4)

optPredARMA11In = nan(6,numObs);

% % Optimal multi-step ahead forecast ARMA(1,1) h=1, h=2, h= 4

for t = 1:tstar-1

    optPredARMA11In(1:4,t+1) = forecast(estMdl1,4,rate1(1:t)');

end

optPredARMA11In = optPredARMA11In([1,2,4,5,6],:);

% % EWMA
optPredARMA11In(4,1) = rate(1);
lambda = 0.94;

for t = 1:tstar-1

    optPredARMA11In(4,t+1)= lambda*rate(t) + (1-lambda)*optPredARMA11In(4,t);

end

% % Flat forecast
 for t = 1:tstar-1

     optPredARMA11In(5,t+1) = rate(t);

 end

optPredARMA11In = optPredARMA11In(:,1:tstar);
optPredARMA11In(:,1) = rate(1,1);

timeIn = time(1:tstar);

figure;
plot(timeIn,rate1)
hold on
plot(timeIn,optPredARMA11In(1,:))
hold on
plot(timeIn,optPredARMA11In(2,:))
hold on
plot(timeIn,optPredARMA11In(3,:))
hold on
plot(timeIn,optPredARMA11In(4,:))
hold on
plot(timeIn,optPredARMA11In(5,:))

legend('rate1','ARMA(1,1) 1 step ahead','ARMA(1,1) 2 step ahead','ARMA(1,1) 4 step ahead','EWMA','Flat')
title('Forecast vs Actual Values-In Sample','Interpreter','latex')
xlabel('Years')


%% Computation of MSE ARMA(1,1)

MSEIn11(1) = mean((optPredARMA11In(1,:) - rate1).^2); %% MSE 1 step ahead forecast
MSEIn11(2) = mean((optPredARMA11In(2,:) - rate1).^2); %% MSE 2 step ahead forecast
MSEIn11(3) = mean((optPredARMA11In(3,:) - rate1).^2); %% MSE 4 step ahead forecast
MSEIn11(4) = mean((optPredARMA11In(4,:) - rate1).^2); %% MSE EWMA
MSEIn11(5) = mean((optPredARMA11In(5,:) - rate1).^2); %% MSE Flat forecast

minMSEIn11 = find(MSEIn11 == min(MSEIn11));

%% In sample forecast ARMA(2,2) (h=1 h=2 h=4)

optPredARMA22In = nan(6,numObs);

% % Optimal multi-step ahead forecast ARMA(1,1) h=1, h=2, h= 4

for t = 2:tstar-1

    optPredARMA22In(1:4,t+1) = forecast(estMdl2,4,rate1(1:t)');

end

optPredARMA22In = optPredARMA22In([1,2,4,5,6],:);

% % EWMA

optPredARMA22In(4,1) = rate(1);
lambda = 0.94;

for t = 1:tstar-1

    optPredARMA22In(4,t+1)= lambda*rate(t) + (1-lambda)*optPredARMA22In(4,t);

end

% % Flat forecast

for t = 1:tstar-1

     optPredARMA22In(5,t+1) = rate1(t);

 end

optPredARMA22In = optPredARMA22In(:,1:tstar);
optPredARMA22In(:,1) = rate1(1,1);
optPredARMA22In(1:3,2) = rate1(1,2);

timeIn = time(1:tstar);

figure;
plot(timeIn,rate1)
hold on
plot(timeIn,optPredARMA22In(1,:))
hold on
plot(timeIn,optPredARMA22In(2,:))
hold on
plot(timeIn,optPredARMA22In(3,:))
hold on
plot(timeIn,optPredARMA22In(4,:))
hold on
plot(timeIn,optPredARMA22In(5,:))

legend('rate1','ARMA(2,2) 1 step ahead','ARMA(2,2) 2 step ahead','ARMA(2,2) 4 step ahead','EWMA','Flat')
title('Forecast vs Actual Values-In Sample Sample','Interpreter','latex')
xlabel('Years')

%% Computation MSE ARMA(2,2) 

MSEIn22(1) = mean((optPredARMA22In(1,:) - rate1).^2); %% MSE 1 step ahead forecast
MSEIn22(2) = mean((optPredARMA22In(2,:) - rate1).^2); %% MSE 2 step ahead forecast
MSEIn22(3) = mean((optPredARMA22In(3,:) - rate1).^2); %% MSE 4 step ahead forecast
MSEIn22(4) = mean((optPredARMA22In(4,:) - rate1).^2); %% MSE EWMA
MSEIn22(5) = mean((optPredARMA22In(5,:) - rate1).^2); %% MSE Flat forecast

minMSEIn22 = find(MSEIn22 == min(MSEIn22));

%% Out-of sample forecast ARMA(1,1)

optPredARMA11Out = nan(6,T);
optPredARMA11Out(5,tstar) = rate(tstar);
 
for t = tstar:numObs

% %     1, 2, 4 step-haead forecast 
    optPredARMA11Out(1:4,t+1) = forecast(estMdl1,4,rate(1:t)');

% % EWMA

    optPredARMA11Out(5,t+1) = rate(t)*lambda + (1-lambda)*optPredARMA11Out(5,t);

% % Flat forecast

    optPredARMA11Out(6,t+1) = rate(t);
end

optPredARMA11Out(3,:) = [];

timeOut = time(tstar+1:numObs);
optPredARMA11Out = optPredARMA11Out(:,tstar+1:numObs);

figure;
plot(timeOut,rate2)
hold on
plot(timeOut,optPredARMA11Out(1,:))
hold on
plot(timeOut,optPredARMA11Out(2,:))
hold on
plot(timeOut,optPredARMA11Out(3,:))
hold on
plot(timeOut,optPredARMA11Out(4,:))
hold on
plot(timeOut,optPredARMA11Out(5,:))

legend('rate2','ARMA(1,1) 1 step ahead','ARMA(1,1) 2 step ahead','ARMA(1,1) 4 step ahead','EWMA','Flat')
title('Forecast vs Actual Values-Out of Sample','Interpreter','latex')
xlabel('Years')

%% Computation MSE ARMA(1,1)

MSEOut11(1) = mean((optPredARMA11Out(1,:) - rate2).^2);
MSEOut11(2) = mean((optPredARMA11Out(2,:) - rate2).^2);
MSEOut11(3) = mean((optPredARMA11Out(3,:) - rate2).^2);
MSEOut11(4) = mean((optPredARMA11Out(4,:) - rate2).^2);
MSEOut11(5) = mean((optPredARMA11Out(5,:) - rate2).^2);

minMSE11Out = find(MSEOut11 == min(MSEOut11));

%% Out of sample forecast ARMA(2,2) 

optPredARMA22Out = nan(6,numObs);
optPredARMA22Out(3,:) = [];
optPredARMA22Out(5,tstar) = rate(tstar);
optPredARMA22Out(5,tstar+1) = rate(tstar+1);

for t = tstar:numObs

% %     1, 2, 4 step-haead forecast 
    optPredARMA22Out(1:4,t+1) = forecast(estMdl2,4,rate(1:t)');

% % EWMA

    optPredARMA22Out(5,t+1) = rate(t)*lambda + (1-lambda)*optPredARMA22Out(5,t);

% % Flat forecast

    optPredARMA22Out(6,t+1) = rate(t);
end

optPredARMA22Out(3,:) = [];
timeOut = time(tstar+1:numObs);
optPredARMA22Out = optPredARMA22Out(:,tstar+1:numObs);

figure;
plot(timeOut,rate2)
hold on
plot(timeOut,optPredARMA22Out(1,:))
hold on
plot(timeOut,optPredARMA22Out(2,:))
hold on
plot(timeOut,optPredARMA22Out(3,:))
hold on
plot(timeOut,optPredARMA22Out(4,:))
hold on
plot(timeOut,optPredARMA22Out(5,:))

legend('rate2','ARMA(2,2) 1 step ahead','ARMA(2,2) 2 step ahead','ARMA(2,2) 4 step ahead','EWMA','Flat')
title('Forecast vs Actual Values-Out of Sample','Interpreter','latex')
xlabel('Years')

%% Computation MSE ARMA(2,2)

MSEOut22(1) = mean((optPredARMA22Out(1,:) - rate2).^2);
MSEOut22(2) = mean((optPredARMA22Out(2,:) - rate2).^2);
MSEOut22(3) = mean((optPredARMA22Out(3,:) - rate2).^2);
MSEOut22(4) = mean((optPredARMA22Out(4,:) - rate2).^2);
MSEOut22(5) = mean((optPredARMA22Out(5,:) - rate2).^2);

minMSE22Out = find(MSEOut22 == min(MSEOut22));

%% Changing tstar 


tstar = 130; 
numObs = length(rate);
rate1 = rate(1:tstar);
rate2 = rate(tstar+1:numObs);

%% Estimation ARMA(1,1) and ARMA(2,2)

mdl1 = arima(1,0,1);
estMdl1 = estimate(mdl1,rate1','Display','off');

mdl2 = arima(2,0,2);
estMdl2 = estimate(mdl2,rate1','Display','off');

%% Forecast In sample of both 2 models


optPredARMA11In = nan(6,numObs);

% % Optimal multi-step ahead forecast ARMA(1,1) h=1, h=2, h= 4

for t = 1:tstar-1

    optPredARMA11In(1:4,t+1) = forecast(estMdl1,4,rate1(1:t)');

end

optPredARMA11In = optPredARMA11In([1,2,4,5,6],:);

% % EWMA
optPredARMA11In(4,1) = rate(1);
lambda = 0.94;

for t = 1:tstar-1

    optPredARMA11In(4,t+1)= lambda*rate(t) + (1-lambda)*optPredARMA11In(4,t);

end

% % Flat forecast
 for t = 1:tstar-1

     optPredARMA11In(5,t+1) = rate(t);

 end

optPredARMA11In = optPredARMA11In(:,1:tstar);
optPredARMA11In(:,1) = rate(1,1);

timeIn = time(1:tstar);

figure;
plot(timeIn,rate1)
hold on
plot(timeIn,optPredARMA11In(1,:))
hold on
plot(timeIn,optPredARMA11In(2,:))
hold on
plot(timeIn,optPredARMA11In(3,:))
hold on
plot(timeIn,optPredARMA11In(4,:))
hold on
plot(timeIn,optPredARMA11In(5,:))

legend('rate1','ARMA(1,1) 1 step ahead','ARMA(1,1) 2 step ahead','ARMA(1,1) 4 step ahead','EWMA','Flat')


optPredARMA22In = nan(6,numObs);

% % Optimal multi-step ahead forecast ARMA(1,1) h=1, h=2, h= 4

for t = 2:tstar-1

    optPredARMA22In(1:4,t+1) = forecast(estMdl2,4,rate1(1:t)');

end

optPredARMA22In = optPredARMA22In([1,2,4,5,6],:);

% % EWMA

optPredARMA22In(4,1) = rate(1);
lambda = 0.94;

for t = 1:tstar-1

    optPredARMA22In(4,t+1)= lambda*rate(t) + (1-lambda)*optPredARMA22In(4,t);

end

% % Flat forecast

for t = 1:tstar-1

     optPredARMA22In(5,t+1) = rate1(t);

 end

optPredARMA22In = optPredARMA22In(:,1:tstar);
optPredARMA22In(:,1) = rate1(1,1);
optPredARMA22In(:,2) = rate1(1,2);

timeIn = time(1:tstar);

figure;
plot(timeIn,rate1)
hold on
plot(timeIn,optPredARMA22In(1,:))
hold on
plot(timeIn,optPredARMA22In(2,:))
hold on
plot(timeIn,optPredARMA22In(3,:))
hold on
plot(timeIn,optPredARMA22In(4,:))
hold on
plot(timeIn,optPredARMA22In(5,:))

legend('rate1','ARMA(2,2) 1 step ahead','ARMA(2,2) 2 step ahead','ARMA(2,2) 4 step ahead','EWMA','Flat')

%% MSE In sample of both 2 models

MSEIn11(1) = mean((optPredARMA11In(1,:) - rate1).^2); %% MSE 1 step ahead forecast
MSEIn11(2) = mean((optPredARMA11In(2,:) - rate1).^2); %% MSE 2 step ahead forecast
MSEIn11(3) = mean((optPredARMA11In(3,:) - rate1).^2); %% MSE 4 step ahead forecast
MSEIn11(4) = mean((optPredARMA11In(4,:) - rate1).^2); %% MSE EWMA
MSEIn11(5) = mean((optPredARMA11In(5,:) - rate1).^2); %% MSE Flat forecast

MSEIn22(1) = mean((optPredARMA22In(1,:) - rate1).^2); %% MSE 1 step ahead forecast
MSEIn22(2) = mean((optPredARMA22In(2,:) - rate1).^2); %% MSE 2 step ahead forecast
MSEIn22(3) = mean((optPredARMA22In(3,:) - rate1).^2); %% MSE 4 step ahead forecast
MSEIn22(4) = mean((optPredARMA22In(4,:) - rate1).^2); %% MSE EWMA
MSEIn22(5) = mean((optPredARMA22In(5,:) - rate1).^2); %% MSE Flat forecast

%% Forecast Out of sample of both models

optPredARMA11Out = nan(6,T);
optPredARMA11Out(5,tstar) = rate(tstar);
 
for t = tstar:numObs

% %     1, 2, 4 step-haead forecast 
    optPredARMA11Out(1:4,t+1) = forecast(estMdl1,4,rate(1:t)');

% % EWMA

    optPredARMA11Out(5,t+1) = rate(t)*lambda + (1-lambda)*optPredARMA11Out(5,t);

% % Flat forecast

    optPredARMA11Out(6,t+1) = rate(t);
end

optPredARMA11Out(3,:) = [];

timeOut = time(tstar+1:numObs);
optPredARMA11Out = optPredARMA11Out(:,tstar+1:numObs);

figure;
plot(timeOut,rate2)
hold on
plot(timeOut,optPredARMA11Out(1,:))
hold on
plot(timeOut,optPredARMA11Out(2,:))
hold on
plot(timeOut,optPredARMA11Out(3,:))
hold on
plot(timeOut,optPredARMA11Out(4,:))
hold on
plot(timeOut,optPredARMA11Out(5,:))

legend('rate2','ARMA(1,1) 1 step ahead','ARMA(1,1) 2 step ahead','ARMA(1,1) 4 step ahead','EWMA','Flat')

optPredARMA22Out = nan(6,numObs);
optPredARMA22Out(3,:) = [];
optPredARMA22Out(5,tstar) = rate(tstar);
optPredARMA22Out(5,tstar+1) = rate(tstar+1);

for t = tstar:numObs

% %     1, 2, 4 step-haead forecast 
    optPredARMA22Out(1:4,t+1) = forecast(estMdl2,4,rate(1:t)');

% % EWMA

    optPredARMA22Out(5,t+1) = rate(t)*lambda + (1-lambda)*optPredARMA22Out(5,t);

% % Flat forecast

    optPredARMA22Out(6,t+1) = rate(t);
end

optPredARMA22Out(3,:) = [];
timeOut = time(tstar+1:numObs);
optPredARMA22Out = optPredARMA22Out(:,tstar+1:numObs);

figure;
plot(timeOut,rate2)
hold on
plot(timeOut,optPredARMA22Out(1,:))
hold on
plot(timeOut,optPredARMA22Out(2,:))
hold on
plot(timeOut,optPredARMA22Out(3,:))
hold on
plot(timeOut,optPredARMA22Out(4,:))
hold on
plot(timeOut,optPredARMA22Out(5,:))

legend('rate2','ARMA(2,2) 1 step ahead','ARMA(2,2) 2 step ahead','ARMA(2,2) 4 step ahead','EWMA','Flat')

%% MSE of both the models

MSEOut11(1) = mean((optPredARMA11Out(1,:) - rate2).^2);
MSEOut11(2) = mean((optPredARMA11Out(2,:) - rate2).^2);
MSEOut11(3) = mean((optPredARMA11Out(3,:) - rate2).^2);
MSEOut11(4) = mean((optPredARMA11Out(4,:) - rate2).^2);
MSEOut11(5) = mean((optPredARMA11Out(5,:) - rate2).^2);

MSEOut22(1) = mean((optPredARMA22Out(1,:) - rate2).^2);
MSEOut22(2) = mean((optPredARMA22Out(2,:) - rate2).^2);
MSEOut22(3) = mean((optPredARMA22Out(3,:) - rate2).^2);
MSEOut22(4) = mean((optPredARMA22Out(4,:) - rate2).^2);
MSEOut22(5) = mean((optPredARMA22Out(5,:) - rate2).^2);

%% Test of presence of autocorrelation among squared residuals

[h_res1_2,pValue1_2] = lbqtest(res1.^2);
[h_res2_2,pValue2_2] = lbqtest(res2.^2);

