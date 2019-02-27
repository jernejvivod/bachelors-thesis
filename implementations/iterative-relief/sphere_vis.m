[x, y, z] = sphere(128);
h = surfl(x, y, z);
set(h, 'FaceAlpha', 0.08);
shading interp;
light('Position',[-1 2 0],'Style','local');
set(h, 'FaceColor', [0 0.1 0.8]);
axis equal; axis square;