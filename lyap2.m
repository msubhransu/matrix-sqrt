function X = lyap2(A, B, C)
%LYAP2  Lyapunov equation solution using eigenvalue decomposition.
%   X = LYAP2(A,C) solves the special form of the Lyapunov matrix
%   equation:
%
%       A*X + X*A' = -C                                                                                                                                 
%                                                                                                                                                       
%   X = LYAP2(A,B,C) solves the general form of the Lyapunov matrix                                                                                     
%   equation:                                                                                                                                           
%                                                                                                                                                       
%       A*X + X*B = -C                                                                                                                                  
%                                                                                                                                                       
%   LYAP2 is faster and generally more accurate than LYAP except when                                                                                   
%   A or B have repeated roots.                                                                                                                         
%                                                                                                                                                       
%   See also DLYAP.                                                                                                                                     
                                                                                                                                                        
%   A.C.W. Grace  10-25-89                                                                                                                              
%   Copyright 1986-2002 The MathWorks, Inc.                                                                                                             
                                                                                                                                                        
[ma,na] = size(A);                                                                                                                                      
[mb,nb] = size(B);                                                                                                                                      
if ((ma ~= na) | (mb ~= nb))                                                                                                                            
     error('Dimensions of A and B do not agree.');                                                                                                      
elseif ma==0,                                                                                                                                           
    X = [];                                                                                                                                             
    return                                                                                                                                              
end                                                                                                                                                     
                                                                                                                                                        
% Perform eigenvalue  decomposition on A and B                                                                                                          
[T,DA]=eig(A);                                                                                                                                          
if nargin==3                                                                                                                                            
    [mc,nc] = size(C);                                                                                                                                  
    if ((mc ~= ma) | (nc ~= mb)), error('Dimensions of C do not agree'); end                                                                            
    [U,DB]=eig(B);                                                                                                                                      
    X=-T*(T\C*U./(DA*ones(ma,mb)+ones(ma,mb)*DB))/U;                                                                                                    
else                                                                                                                                                    
% Shortcut for AX+XA'+C=0
% Note B is really C for two input arguments
    IT=inv(T);
    DA=DA*ones(ma,ma);
    X=-T*(IT*B*IT.'./(DA+DA.'))*T.';                                                                                                                    
    C=0;                                                                                                                                                
end                                                                                                                                                     
                                                                                                                                                        
% ignore complex part if real inputs (better be small)                                                                                                  
    if ~(any(any(imag(A))) | any(any(imag(B))) | any(any(imag(C))))                                                                                     
        X = real(X);                                                                                                                                    
    end                                                                                                                                                 

