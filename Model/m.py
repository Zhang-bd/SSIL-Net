import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Multi_Scale(nn.Module):
    def __init__(self, x_in_c, y_in_c, out_c):
        super(Multi_Scale, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(x_in_c, out_c, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(y_in_c, out_c, kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU()
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        # self.conv3 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=5, stride=1, padding=2)
        # self.conv7 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=7, stride=1, padding=3)
        self.conv11 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=out_c * 2, out_channels=out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=out_c*2, out_channels=out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )
        # self.fre = Freprocess2(x_in_c, y_in_c, out_c)

    def forward(self, x, y):
        # frequency
        # f = self.fre(x,y)

        x = self.conv1(x)
        y = self.conv2(y)

        x = torch.cat((x, y), dim=1)
        x = self.conv_block(x)
        # x = x + f

        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        #
        y_avg = torch.mean(y, dim=1, keepdim=True)
        y_max, _ = torch.max(y, dim=1, keepdim=True)
        # ch_avg = self.global_avg_pool(y)
        # ch_max = self.global_max_pool(y)
        #
        x1 = torch.sigmoid(x_avg + x_max) * x
        x = self.conv11(x) + x1
        #
        y1 = torch.sigmoid(y_avg + y_max) * y
        y = self.conv22(y) + y1

        x = torch.cat((x, y), dim=1)

        # x = torch.sigmoid(ch_max+ch_avg) * x + x
        x = self.conv(x)
        return x


class Freprocess2(nn.Module):
    def __init__(self, x_c, y_c, channels):
        super(Freprocess2, self).__init__()
        self.pre1 = nn.Conv2d(y_c, channels, 1, 1, 0)
        self.pre2 = nn.Conv2d(x_c, channels, 1, 1, 0)
        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, panf, msf):
        _, _, H, W = msf.shape
        msF = torch.fft.rfft2(self.pre1(msf) + 1e-8, norm='backward')
        panF = torch.fft.rfft2(self.pre2(panf) + 1e-8, norm='backward')
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        panF_amp = torch.abs(panF)
        panF_pha = torch.angle(panF)

        amp_fuse = self.amp_fuse(panF_amp - msF_amp) + panF_amp

        pha_fuse = self.pha_fuse(panF_pha - msF_pha) + panF_pha

        # amp_fuse = self.amp_fuse(torch.cat([msF_amp, panF_amp], 1))
        # pha_fuse = self.pha_fuse(torch.cat([msF_pha, panF_pha], 1))

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8
        out = torch.complex(real, imag) + 1e-8
        out = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return self.post(out)




class SpeFE(nn.Module):
    def __init__(self, in_channels=1, out_channels=28):
        super(SpeFE, self).__init__()
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.spc1 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(out_channels), )

        self.spc2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.LeakyReLU(inplace=True), )

        self.bn2 = nn.BatchNorm3d(out_channels)
        self.spc3 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm3d(out_channels), )

        self.spc4 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.LeakyReLU(inplace=True), )

        self.bn3 = nn.BatchNorm3d(out_channels)

        self.last = nn.Sequential(
            nn.Conv3d(out_channels, 1, kernel_size=(7, 1, 1), padding=(3, 0, 0), bias=False),
            nn.LeakyReLU(inplace=True), )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.s1(x)
        # print(x.shape) stride
        x1 = self.bn1(F.leaky_relu(x))
        x = self.spc1(x1)
        x = self.bn2(self.spc2(x))
        x2 = F.leaky_relu(x + x1)

        x = self.spc3(x2)
        x = self.bn3(self.spc4(x))
        x2 = F.leaky_relu(x + x2)
        x2 = self.last(x2)
        x = x2.squeeze(1)

        return x


class ResSPA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResSPA, self).__init__()

        self.spa1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                  nn.LeakyReLU(inplace=True),
                                  nn.BatchNorm2d(in_channels), )

        self.spa2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
                                  nn.LeakyReLU(inplace=True),
                                  nn.BatchNorm2d(in_channels),)
        self.spa3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3),
                                  nn.LeakyReLU(inplace=True),
                                  nn.BatchNorm2d(in_channels),)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*3,out_channels*2,3,1,1),
            nn.Conv2d(in_channels * 2, out_channels , 3, 1, 1),
            nn.ReLU()

        )
        # self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        x1 = self.spa1(input)
        x2 = self.spa2(input)
        x3 = self.spa3(input)

        x = torch.cat((x1,x2,x3),dim=1)
        x = self.conv(x)
        # out = self.spa1(input)
        # out = self.bn2(self.spa2(out))

        return F.leaky_relu(x + input)


class Attention(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class FuseBlock7(nn.Module):
    def __init__(self, channels):
        super(FuseBlock7, self).__init__()
        self.fre = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fre_att = Attention(dim=channels)
        # self.spa_att = Attention(dim=channels)
        self.fuse = nn.Sequential(nn.Conv2d(2 * channels, channels, 3, 1, 1),
                                  nn.Conv2d(channels, channels, 3, 1, 1),
                                  nn.LeakyReLU())

        self.pointconv = nn.Sequential(
            nn.Conv2d(channels,channels,1),
            nn.LeakyReLU(),
            nn.Conv2d(channels,channels,3,1,1)
        )
        self.depthconv = nn.Sequential(
            nn.Conv2d(channels,channels,kernel_size=3,padding=1,groups=channels),
            nn.LeakyReLU(),
            nn.Conv2d(channels,channels,3,1,1)
        )
        self.scale1 = nn.Parameter(
            1e-5 * torch.ones((1, channels, 1, 1)),
            requires_grad=True
        )
        self.scale2= nn.Parameter(
            1e-5 * torch.ones((1, channels, 1, 1)),
            requires_grad=True
        )
    def forward(self, spa, fre):
        fre = self.fre(fre)
        spa = self.spa(spa)
        fre = self.fre_att(fre, spa) + fre
        spa = self.fre_att(spa, fre) + spa

        fuse = self.fuse(torch.cat((fre, spa), 1))

        spa = self.depthconv(fuse - spa) #* self.scale1
        fre = self.pointconv(fuse - fre) #* self.scale2
        fuse = spa + fre + fuse
        """
        fuse = fuse * self.spatial(spa) + fuse
        fuse = self.conv3(fuse)
        avg_pool = torch.mean(fre, dim=[2,3], keepdim=True)
        avg_pool = self.fc2(F.relu(self.fc1(avg_pool)))

        fuse = torch.sigmoid(avg_pool) * fuse + fuse
        """
        return fuse


class SuperResolution(nn.Module):
    def __init__(self, upscale_factor, in_channels=1, out_channels=1):
        super(SuperResolution, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)  # �������ͼ��H��W����
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # �������ͼ��H��W����
        # self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)  # �������ͼ��H��W����
        self.conv4 = nn.Conv2d(64, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)  # �������ͼ��H��W����
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)  # �������ͼ��H��W��Ϊr��H��r��W

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return self.pixel_shuffle(x)

class Model(nn.Module):
    def __init__(self, h, m, p, hidden=64):
        super(Model, self).__init__()
        self.m = m
        self.up2 = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=True)
        self.m2h = nn.Conv2d(m, h, kernel_size=1)
        self.p2h = nn.Conv2d(p, h, kernel_size=1)
        self.up16 = nn.Upsample(scale_factor=16, mode='bicubic', align_corners=True)

        self.srh4 = SuperResolution(4,h,h)
        self.srm4 = SuperResolution(4,m, m)
        self.hidden4 = SuperResolution(4, h, h)

        self.mls1 = Multi_Scale(p, m, hidden)
        self.mls2 = Multi_Scale(m, h, h)
        self.mls3 = Multi_Scale(hidden, hidden, hidden)

        # self.spa0 = Spa(64)
        self.last_conv = nn.Sequential(
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.Conv2d(hidden * 2, h, kernel_size=3, padding=1)
        )
        # self.conv1_1 = nn.Conv2d(h, hidden, kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv0 = nn.Sequential(
            nn.Conv2d(hidden+1, hidden, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        # self.spe1 = SpeFE()
        self.spa1 = ResSPA(hidden, hidden)
        self.spa2 = ResSPA(hidden, hidden)
        self.spa3 = ResSPA(hidden, hidden)

        # self.spa4 = ResSPA(hidden, hidden)


        self.fus1 = FuseBlock7(hidden)
        self.fus2 = FuseBlock7(hidden)
        self.fus3 = FuseBlock7(hidden)
        self.fus4 = FuseBlock7(hidden)


        self.hid = nn.Conv2d(h,hidden,kernel_size=1)

        self.tt1 = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden * 2, hidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Conv2d(hidden,hidden,1)
        )
    def forward(self, pan, msi, hsi):
        # print('hsi',hsi.shape)
        hsr = self.srh4(hsi) # 1->4
        msr = self.srm4(msi) # 4->16

        h16 = self.up16(hsi) # 1->16
        msr = self.up4(msi) + msr # 16
        hsi = self.up4(hsi) + hsr # 4

        pm = self.mls1(pan, msr)
        mh = self.mls2(msi, hsi)  # 4
        mh = self.hidden4(mh) +  h16 #16
        mh = self.hid(mh)

        x = self.mls3(pm, mh)  # spatial

        y = torch.cat((x, mh), dim=1)  # spectral
        y = self.conv1(y)

        x = self.conv0(torch.cat((x , pan),dim=1))  # enchance spatial

        # y = self.spe1(y)

        x = self.spa1(x)
        x = self.spa2(x)
        x = self.spa3(x)

        s1 = self.fus1(x, y)
        s2 = self.fus2(x, s1)
        s3 = self.fus3(x, s2)
        s4 = self.fus4(x, s3)

        t1 = torch.cat((s1, s2, s3, s4), dim=1)
        t1 = self.tt1(t1)

        out = self.last_conv(t1) + h16
        return out


if __name__ == '__main__':
    h = 96

    x = torch.randn(1, 1, h, h)
    y = torch.randn(1, 4, h // 4, h // 4)
    z = torch.randn(1, 128, h // 16, h // 16)
    # t = torch.randn(1,64, 64, 64)
    # x,_ = torch.max(x,dim=1,keepdim=True)
    # model = Multi_Scale(3,3,64)
    model = Model(128, 4, 1)
    # model = M1()
    # out = model(t)
    # print(out.shape)
    out = model(x, y, z)
    print(out.shape)
    print('# parameters:', sum(param.numel() for param in model.parameters()))
    parameter_nums = sum(p.numel() for p in model.parameters())
    print("Model size:", str(float(parameter_nums / 1e6)) + 'M')




    from thop import profile, clever_format
    import torch
    import time
    
    model = Model(128,4,1).cuda()
    model.eval()
    
    pan = torch.randn(1,1,96,96).cuda()
    msi = torch.randn(1,4,24,24).cuda()
    hsi = torch.randn(1,128,6,6).cuda()
    
    # FLOPs & Params
    flops, params = profile(
        model,
        inputs=(pan,msi,hsi),
        verbose=False
    )
    
    flops, params = clever_format([flops, params], "%.3f")
    
    print("FLOPs:", flops)
    print("Params:", params)
    
    # Runtime
    with torch.no_grad():
    
        for _ in range(50):
            _ = model(pan,msi,hsi)
    
        torch.cuda.synchronize()
    
        N = 100
    
        start = time.time()
    
        for _ in range(N):
            _ = model(pan,msi,hsi)
    
        torch.cuda.synchronize()
    
        end = time.time()
    
    print(f'Runtime: {(end-start)/N*1000:.3f} ms')
