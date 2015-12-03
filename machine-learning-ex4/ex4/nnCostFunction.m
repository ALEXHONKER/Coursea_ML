function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

X=[ones(m,1),X];
for i = 1:m
	for j=1:size(Theta2,1)
		h=sigmoid(Theta2*([1;sigmoid(Theta1*X(i,:)')]));
		yy=zeros(1,size(Theta2,1));
		if y(i)==0
			yy(10)=1;
		else
		yy(y(i))=1;
		end
		J=J+(-1*(yy*log(h))- (1.-yy)*(log(1.-h)));
	end
end
J=J/m;
J=J/size(Theta2,1);
th=0;
for i=1:size(Theta1,1)
	for j=2:size(Theta1,2)
		th=th+Theta1(i,j)**2;	
	end
end
for i=1:size(Theta2,1)
	for j=2:size(Theta2,2)
		th=th+Theta2(i,j)**2;
	end
end
th=th*lambda/(2*m);
J=J+th;
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
delta1=zeros(size(Theta1));
delta2=zeros(size(Theta2));
for t=1:m
	a1=X(t,:);
	z2=Theta1*a1';
	a2=sigmoid(z2);
	a2=[1;a2];
	newa2=a2;
	z3=Theta2*newa2;
	a3=sigmoid(z3);
	yy=zeros(1,size(Theta2,1));
	if y(t)==0
		yy(10)=1;
	else
	yy(y(t))=1;
	end
	th3=a3.-yy';
	th2=Theta2'*th3.*[1;sigmoidGradient(z2)];
	delta1=delta1+th2(2:size(th2))*a1;
	delta2=delta2+th3*a2';
end
Theta1_grad=delta1./m;
Theta2_grad=delta2./m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


for i=1:size(Theta1,1)
	Theta1_grad(i,1)=delta1(i,1)/m;
	for j=2:size(Theta1,2)
		Theta1_grad(i,j)=delta1(i,j)/m+Theta1(i,j)*lambda/m;
	end
end
for i=1:size(Theta2,1)
        Theta2_grad(i,1)=delta2(i,1)/m;
        for j=2:size(Theta2,2)
                Theta2_grad(i,j)=delta2(i,j)/m+Theta2(i,j)*lambda/m;
        end
end
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
