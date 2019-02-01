clear all 
close all
%loading images
mona = load('D:\Sweta\Ph.D\Course_work\1st_year\CNS_assignments\BT6270_Assignment_4\mona.txt');
cat = load('D:\Sweta\Ph.D\Course_work\1st_year\CNS_assignments\BT6270_Assignment_4\cat.txt');
%change cat in binary format
cat1 = reshape(cat,9000,1);

cat1(find(cat1>0),:) = 1;

cat1(find(cat1<=0),:) = -1;

cat = reshape(cat1,90,100);

ball = load('D:\Sweta\Ph.D\Course_work\1st_year\CNS_assignments\BT6270_Assignment_4\ball.txt');

x(:,:,1) = ball;
x(:,:,2) = cat;
x(:,:,3) = mona;

%prepare traindata
train(1,:) = reshape(x(:,:,1), 1, 9000);
train(2,:) = reshape(x(:,:,2), 1, 9000);
train(3,:) = reshape(x(:,:,3), 1, 9000);

prompt = 'Enter the question number (1 or 2 or 3):';
ques = input(prompt)

if (ques == 1)
    %path for result
    path = 'D:\Sweta\Ph.D\Course_work\1st_year\CNS_assignments\BT6270_Assignment_4\Hebbian_Learning\Results\Result_1_b';
    %prepare trigger images for ques 1
    mona(20:50,20:50)=-1;
    cat(20:50,20:50)=-1;
    ball(20:50,20:50)=-1;

    y(:,:,1) = ball;
    y(:,:,2) = cat;
    y(:,:,3) = mona;

    trigger(1,:) = reshape(y(:,:,1), 1, 9000);
    trigger(2,:) = reshape(y(:,:,2), 1, 9000);
    trigger(3,:) = reshape(y(:,:,3), 1, 9000);
else
    %path for result
    path = 'D:\Sweta\Ph.D\Course_work\1st_year\CNS_assignments\BT6270_Assignment_4\Hebbian_Learning\Results\Result_2_a_b_c_3_a_b';
    %prepare trigger images for question 2 or 3
    z = zeros(90,100);

    z(40:50,40:50) = ball(40:50,40:50);
    y(:,:,1) = z;

    z(40:50,40:50) = cat(40:50,40:50);
    y(:,:,2) = z;

    z(40:50,40:50) = mona(40:50,40:50);
    y(:,:,3) = z;

    trigger(1,:) = reshape(y(:,:,1), 1, 9000);
    trigger(2,:) = reshape(y(:,:,2), 1, 9000);
    trigger(3,:) = reshape(y(:,:,3), 1, 9000);
end
%initialize rms array
rms = zeros(3,9000);

% Initialize weight matrix
W = zeros(size(x,1),size(x,2));

% Calculate weight matrix = learning
for i = 1:1:size(x,1)*size(x,2)
    for j = 1:1:9000
        weight = 0;
        if (i ~= j)
            for n = 1:1:size(x,3)
                weight = train(n,i) .* train(n,j) + weight;                
            end
        end
        W(i,j) = weight;
    end
end


if (ques == 3)
    
    % give % of weights to be 0
    prompt = 'Enter (25 or 50 or 80)% of weights to be 0:';
    X = input(prompt);
    if(X==25)
        %path for result
        path = 'D:\Sweta\Ph.D\Course_work\1st_year\CNS_assignments\BT6270_Assignment_4\Hebbian_Learning\Results\Result_3_c\Result_X_25_1';
    end
    if(X==50)
        %path for result
        path = 'D:\Sweta\Ph.D\Course_work\1st_year\CNS_assignments\BT6270_Assignment_4\Hebbian_Learning\Results\Result_3_c\Result_X_50_1';
    end
    if(X == 80)
        %path for result
    path = 'D:\Sweta\Ph.D\Course_work\1st_year\CNS_assignments\BT6270_Assignment_4\Hebbian_Learning\Results\Result_3_c\Result_X_80_1';
    end
  
    %save the previous weights
    saveas(imshow(W),fullfile(path,'WeightBefore.jpg'));
    
    % X % of weights is 0
    X_w = X * size(W,1) * size(W,2) * 0.01;
    W_indices = randperm(81000000, X_w);
    W = reshape(W, 1, 81000000);
    W(W_indices(1,:)) = 0;
    W = reshape(W, 9000, 9000);
    saveas(imshow(W),fullfile(path,'WeightAfter.jpg'));
end

% Retrieve
% At each iteration retrieve image is same, that means error will not be
% reduced
for n = 1:1:size(y,3)
    for iter = 1:1:50
        % Generate random element for the asynchronous correction
%         i = randi([1 size(x,1)*size(x,2)],1,1);
        for i = 1: 1: size(x,1)*size(x,2)
        sum = 0;
        for j = 1:1:size(x,1)*size(x,2)
            sum = sum + W(i, j) * trigger(n,j);
        end
        
        % Therehold
        retrieved(n,i) = sign(sum);
        end
    
        rms(n,iter) = sqrt((mean((train(n,:)-retrieved(n,:)).^2))); 
    end
end

%plotting and saving the patch and retrieved images
for k = 1:3
saveas(imshow(y(:,:,k)),fullfile(path,['trigger' num2str(k) '.jpg']));
saveas(imshow(reshape(retrieved(k,:), 90, 100)),fullfile(path,['retrieve' num2str(k) '.jpg']));
end

%plotting and saving the rms 
figure(1)
plot(rms(1,1:iter),'r-','LineWidth',1.5);
xlim([0,iter])
hold on;
plot(rms(2,1:iter),'b-','LineWidth',1.5);
plot(rms(3,1:iter),'g-','LineWidth',1.5);
legend('Ball','Cat','Monalisa');
xlabel('Time')
ylabel('RMS')
title('Time vs. RMS');
hold off;
saveas(figure(1),fullfile(path,'Plot.jpg'));
