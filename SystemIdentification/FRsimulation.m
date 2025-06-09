% authors: Lucas Colomera, Rodrigo W. Pisaia, Pedro G. de Paula 

function yhat = FRsimulation(Y,U,na,nb,th_hat)
U = U(:);
Y = Y(:);
p = max(na,nb)+1; 
N = length(Y);

yhat = zeros(N,1);
yhat(1:p-1) = Y(1:p-1);

for k = p:N
    auxY = [yhat(k-p+1:k-1); 0];
    auxU = [U(k-p+1:k-1); 0];
    [fr_input, ~] = matReg(auxY, auxU, na,nb);
    yhat(k) = fr_input*th_hat;
end
yhat = yhat(p:end);


end