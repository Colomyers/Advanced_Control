% authors: Lucas Colomera, Rodrigo W. Pisaia, Pedro G. de Paula 

function [Phi,Y] = matReg(y, u, na, nb)
y = y(:);
u = u(:);
p = max(na,nb) +1;
Ny = length(y); 
Nu = length(u);


if Ny ~= Nu 
    error("Mismatch in vectors y and u dimension")
   
end

Y = y(p:end); % target vector

Phi = zeros(Ny-p+1,na+nb); % regression matrix


for i = 1:na
   Phi(:,i) = y(p-i: Ny-i);
end

for i = 1:nb
    Phi(:,i+na) = u(p-i: Ny-i);
end 

end