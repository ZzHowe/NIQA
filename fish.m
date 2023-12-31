function dst = fish(org_img)



 
lvl = 3;
[bands] = dwt_cdf97(org_img,lvl);

alpha = [4 2 1];

ss3 = ssq(bands{3});

ss2 = ssq(bands{2});

ss1 = ssq(bands{1});

ss = [ss1 ss2 ss3];

dst = max(sum(ss.*alpha));

function ss = ssq(bands)

alpha = 0.8;

lh_img = bands{1}.^2;
hl_img = bands{2}.^2;
hh_img = bands{3}.^2;

E_lh = log10(1+mean(lh_img(:)));
E_hl = log10(1+mean(hl_img(:)));
E_hh = log10(1+mean(hh_img(:)));

ss = alpha*E_hh + (1-alpha) *(E_lh + E_hl)/2;
