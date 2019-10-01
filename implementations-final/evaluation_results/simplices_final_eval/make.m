I1 = imread('1.png');
I2 = imread('2.png');
I3 = imread('3.png');
I4 = imread('4.png');
I5 = imread('5.png');
I6 = imread('6.png');
I7 = imread('7.png');
I8 = imread('8.png');
I9 = imread('9.png');


I10 = imread('10.png');
I11 = imread('11.png');
I12 = imread('12.png');
I13 = imread('13.png');
I14 = imread('14.png');


I1 = I1(170:end-100, 150:end-140, :);
I2 = I2(170:end-100, 150:end-50, :);
I3 = I3(170:end-100, 150:end-130, :);
I4 = I4(170:end-100, 150:end-145, :);
I5 = I5(170:end-100, 150:end-50, :);
I6 = I6(170:end-100, 150:end-135, :);
I7 = I7(170:end-100, 150:end-145, :);
I8 = I8(170:end-100, 150:end-45, :);
I9 = I9(170:end-100, 150:end-130, :);
I10 = I10(170:end-100, 150:end-130, :);
I11 = I11(170:end-100, 150:end-45, :);
I12 = I12(170:end-100, 150:end-130, :);
I13 = I13(170:end-100, 150:end-45, :);
I14 = I14(170:end-100, 150:end-130, :);


init_pad = 255*ones(size(I1, 1), 10, 3);

line1 = [init_pad I1 I2 init_pad];
line2 = [I3 I4 255*ones(size(I3, 1), 1, 3)];
line3 = [I5 I6 255*ones(size(I3, 1), 1, 3)];
line4 = [I7 I8];
line5 = [I9 I10];
line6 = [I11 I12 255*ones(size(I3, 1), 1, 3)];
line7 = [I13 I14 255*ones(size(I3, 1), 1, 3)];

padding1 = 255*ones(size(line1, 1), (size(line1, 2) - size(line2, 2))/2, 3);
line2 = [padding1 line2 padding1];

padding2 = 255*ones(size(line1, 1), (size(line1, 2) - size(line3, 2))/2, 3);
line3 = [padding2 line3 padding2];

padding3 = 255*ones(size(line1, 1), (size(line1, 2) - size(line5, 2))/2, 3);
line5 = [padding3, line5, padding3];

padding4 = 255*ones(size(line1, 1), (size(line1, 2) - size(line6, 2))/2, 3);
line6 = [padding4 line6 padding4];

padding5 = 255*ones(size(line1, 1), (size(line1, 2) - size(line7, 2))/2, 3);
line7 = [padding5 line7 padding5];

padding6 = 255*ones(size(line4, 1), (size(line1, 2) - size(line4, 2))/2, 3);
line4 = [padding6 line4 padding6];
