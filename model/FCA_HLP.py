import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F

from model.ASPP import ASPP
from model.PSPNet import OneModel as PSPNet
from model.feature import extract_feat_res, extract_feat_vgg
from functools import reduce
from operator import add
def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def get_gram_matrix(fea):
    b, c, h, w = fea.shape
    fea = fea.reshape(b, c, h * w)  # C*N
    fea_T = fea.permute(0, 2, 1)  # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T) / (torch.bmm(fea_norm, fea_T_norm) + 1e-7)  # C*C
    return gram

def cross_attn(query, key, value): # generate pixel-level prototype
    Bs, C, H, W = query.shape
    q = query.clone().view(Bs, C, -1).permute(0, 2, 1)
    k = key.clone().view(Bs, C, -1)
    v = value.clone().view(Bs, C, -1).permute(0, 2, 1)
    attn = (q @ k) / (C ** 0.5)  # [Bs, N, N]
    num = torch.where(attn != 0, 1, 0)  # [B,N,N]
    num = num.sum(dim=-1, keepdim=True)  # [B,N,1]
    attn_weight = attn.sum(dim=-1, keepdim=True)  # [B,N,1]
    attn_weight = attn_weight / (num + 1e-7)  # [B,N,1]
    attn_weight = (attn_weight - attn_weight.min(1)[0].unsqueeze(1)) / (
            attn_weight.max(1)[0].unsqueeze(1) - attn_weight.min(1)[0].unsqueeze(1) + 1e-7)
    attn = torch.tensor(attn, dtype=torch.double)
    attn = torch.where(attn == 0, -1e7, attn)  # [B,N,N]
    attn = torch.tensor(attn, dtype=torch.float)
    attn = attn.softmax(dim=-1)
    x = (attn_weight * (attn @ v))
    x = x.permute(0, 2, 1).view(Bs, C, H, W)
    return x

def FCA(query_feats, support_feats, serial, blocks, stack_id): # Multi-Layer Pixel-Level Feature Cross-Activation
    eps = 1e-7
    corr_masks = []
    for i in range(len(serial)):
        b, c, h, w = query_feats[i].shape
        queryIJ = query_feats[i].reshape(b, c, h * w)#b,c,n
        query_norm = queryIJ.norm(2, 1, True)
        s_b, s_c, s_h, s_w = support_feats[i].shape
        supIJ = support_feats[i].reshape(s_b, s_c, s_h * s_w).permute(0, 2, 1)#b,n,c
        supp_norm = supIJ.norm(2, 2, True)
        corr_mask = torch.bmm(supIJ, queryIJ) / (torch.bmm(supp_norm, query_norm) + eps)#b,n,n
        corr_mask = corr_mask.view(b, s_h * s_w, h, w)
        corr_mask = blocks[i](corr_mask)
        corr_masks.append(corr_mask)#n,b,1,h,w

    mask_corr_l4 = torch.cat(corr_masks[-stack_id[0]:], dim=1).contiguous()#b,n,h,w
    mask_corr_l3 = torch.cat(corr_masks[-stack_id[1]:-stack_id[0]], dim=1).contiguous()
    mask_corr_l2 = torch.cat(corr_masks[-stack_id[2]:-stack_id[1]], dim=1).contiguous()
    return [mask_corr_l4, mask_corr_l3, mask_corr_l2]

def prior_mask_optimization(query_feats, mask):
    corr_query_list = []
    for i, query_feat in enumerate(query_feats):
        temp_mask = F.interpolate(mask, query_feat.size()[-2:], mode='bilinear', align_corners=True)
        cosine_eps = 1e-7
        q = query_feat
        s = query_feat * temp_mask
        bsize, ch_sz, sp_sz, _ = q.size()[:]

        tmp_query = q
        tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s
        tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)
        tmp_supp = tmp_supp.permute(0, 2, 1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

        similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
        similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)
        similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
        corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
        corr_query = F.interpolate(corr_query, size=query_feat.size()[-2:], mode='bilinear', align_corners=True)
        corr_query_list.append(corr_query)
    posterior_masks = torch.cat(corr_query_list, 1)
    return posterior_masks

class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.layers = args.layers
        self.mask_size = args.mask_size
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.vgg = args.vgg
        sizes = args.sizes
        self.dataset = args.data_set
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.alpha = torch.nn.Parameter(torch.Tensor([0.8, 0.2]))
        self.drop = 0.1
        self.pretrained = True
        self.classes = 2
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60

        assert self.layers in [50, 101, 152]

        backbone_str = 'vgg' if args.vgg else 'resnet' + str(args.layers)

        if backbone_str == 'vgg':
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
            self.nsimlairy = [1, 3, 3]
            self.size = reduce(add, [[sizes[i]] * x for i, x in enumerate(nbottlenecks[-3:])])
            self.serial = list(range(7, 14))
            self.mask_channel = [32, 16, 8]
            self.concat_channel = 56
        elif backbone_str == 'resnet50':
            self.feat_ids = list(range(3, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
            self.nsimlairy = [3, 6, 4]
            self.size = reduce(add, [[sizes[i]] * x for i, x in enumerate(nbottlenecks[-3:])])
            self.serial = list(range(3, 16))
            self.mask_channel = [32, 16, 8]
            self.concat_channel = 104
        elif backbone_str == 'resnet101':
            self.feat_ids = list(range(3, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
            self.nsimlairy = [3, 23, 4]
        else:
            raise Exception('Unavailable backbone: %s' % backbone_str)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]

        PSPNet_ = PSPNet(args)
        weight_path = 'initmodel/PSPNet/{}/split{}/{}/best.pth'.format(args.data_set, args.split, backbone_str)
        new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']

        try:
            PSPNet_.load_state_dict(new_param)
        except RuntimeError:  # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            PSPNet_.load_state_dict(new_param)

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = PSPNet_.layer0, PSPNet_.layer1, PSPNet_.layer2, PSPNet_.layer3, PSPNet_.layer4

        # Base Learner
        self.learner_base = nn.Sequential(PSPNet_.ppm, PSPNet_.cls)

        # Meta Learner
        self.corr_blocks = nn.ModuleList()
        for i in range(len(self.size)):
            self.corr_blocks.append(nn.Sequential(
                nn.Conv2d(self.size[i] * self.size[i], self.mask_channel[0], kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.mask_channel[0], self.mask_channel[1], kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.mask_channel[1], self.mask_channel[2], kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.mask_channel[2], self.mask_channel[2], kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))
        reduce_dim = 256
        self.low_fea_id = args.low_fea[-1]
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5))
        mask_add_num = 1 + 1 + self.concat_channel
        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim * 3 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.ASPP_meta = ASPP(reduce_dim)
        self.res1_meta = nn.Sequential(
            nn.Conv2d(reduce_dim * 5, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))
        self.res2_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True))
        self.cls_meta = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(reduce_dim, self.classes, kernel_size=1))

        # Gram and Meta
        self.gram_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.gram_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.gram_merge.weight))

        # Learner Ensemble
        self.cls_merge = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.cls_merge.weight = nn.Parameter(torch.tensor([[1.0], [0.0]]).reshape_as(self.cls_merge.weight))

        # K-Shot Reweighting
        if args.shot > 1:
            self.kshot_trans_dim = args.kshot_trans_dim
            if self.kshot_trans_dim == 0:
                self.kshot_rw = nn.Conv2d(self.shot, self.shot, kernel_size=1, bias=False)
                self.kshot_rw.weight = nn.Parameter(torch.ones_like(self.kshot_rw.weight) / args.shot)
            else:
                self.kshot_rw = nn.Sequential(
                    nn.Conv2d(self.shot, self.kshot_trans_dim, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.kshot_trans_dim, self.shot, kernel_size=1))

        self.sigmoid = nn.Sigmoid()
    def get_optim(self, model, args, LR):
        if args.shot > 1:
            optimizer = torch.optim.SGD(
                [
                    {'params': model.alpha},
                    {'params': model.corr_blocks.parameters()},
                    {'params': model.down_query.parameters()},
                    {'params': model.down_supp.parameters()},
                    {'params': model.init_merge.parameters()},
                    {'params': model.ASPP_meta.parameters()},
                    {'params': model.res1_meta.parameters()},
                    {'params': model.res2_meta.parameters()},
                    {'params': model.cls_meta.parameters()},
                    {'params': model.cls_merge.parameters()},
                    {'params': model.gram_merge.parameters()},
                    {'params': model.kshot_rw.parameters()},
                ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(
                [
                    {'params': model.alpha},
                    {'params': model.corr_blocks.parameters()},
                    {'params': model.down_query.parameters()},
                    {'params': model.down_supp.parameters()},
                    {'params': model.init_merge.parameters()},
                    {'params': model.ASPP_meta.parameters()},
                    {'params': model.res1_meta.parameters()},
                    {'params': model.res2_meta.parameters()},
                    {'params': model.cls_meta.parameters()},
                    {'params': model.cls_merge.parameters()},
                    {'params': model.gram_merge.parameters()},
                ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        return optimizer

    def freeze_modules(self, model):
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.learner_base.parameters():
            param.requires_grad = False

    def forward(self, x, s_x, s_y, y_m, y_b, cat_idx=None):
        x_size = x.size()
        bs = x_size[0]
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        with torch.no_grad():
            query_feats, query_backbone_layers = self.extract_feats(x,
                                                                    [self.layer0, self.layer1, self.layer2, self.layer3,
                                                                     self.layer4], self.feat_ids, sum(self.nsimlairy),
                                                                    self.bottleneck_ids, self.lids)

        if self.vgg:
            query_feat = F.interpolate(query_backbone_layers[2],
                                       size=(query_backbone_layers[3].size(2), query_backbone_layers[3].size(3)), \
                                       mode='bilinear', align_corners=True)
            query_feat = torch.cat([query_backbone_layers[3], query_feat], 1)
        else:
            query_feat = torch.cat([query_backbone_layers[3], query_backbone_layers[2]], 1)

        query_feat = self.down_query(query_feat)

        supp_pro_list = []
        final_supp_list = []
        mask_list = []
        supp_feat_list = []
        supp_map_pro_list = []
        for i in range(self.shot):
            mask = (s_y[:, i, 1, :, :] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_0 = self.layer0(s_x[:, i, 1, :, :, :])
                supp_feat_1 = self.layer1(supp_feat_0)
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear',
                                     align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2), supp_feat_3.size(3)),
                                                mode='bilinear', align_corners=True)
                    supp_feat = torch.cat([supp_feat_2, supp_feat_3], 1)
                else:
                    supp_feat = torch.cat([supp_feat_2, supp_feat_3], 1)
                mask_down = F.interpolate(mask,
                                          size=(supp_feat.size(2), supp_feat.size(3)),
                                          mode='bilinear', align_corners=True)
            supp_feat = self.down_supp(supp_feat)
            supp_masked = supp_feat * mask_down
            supp_pro = cross_attn(query_feat, supp_masked, supp_masked) # generate pixel-level prototype
            supp_map_pro = Weighted_GAP(supp_feat, mask_down)
            supp_map_pro_list.append(supp_map_pro)
            supp_pro_list.append(supp_pro)
            supp_feat_list.append(supp_feat_2)

        # K-Shot Reweighting
        que_gram = get_gram_matrix(query_backbone_layers[2])  # [bs, C, C] in (0,1)
        norm_max = torch.ones_like(que_gram).norm(dim=(1, 2))
        est_val_list = []
        for supp_item in supp_feat_list:
            supp_gram = get_gram_matrix(supp_item)
            gram_diff = que_gram - supp_gram
            est_val_list.append((gram_diff.norm(dim=(1, 2)) / norm_max).reshape(bs, 1, 1, 1))  # norm2
        est_val_total = torch.cat(est_val_list, 1)  # [bs, shot, 1, 1]
        if self.shot > 1:
            val1, idx1 = est_val_total.sort(1)
            val2, idx2 = idx1.sort(1)
            weight = self.kshot_rw(val1)
            weight = weight.gather(1, idx2)
            weight_soft = torch.softmax(weight, 1)
        else:
            weight_soft = torch.ones_like(est_val_total)
        est_val = (weight_soft * est_val_total).sum(1, True)  # [bs, 1, 1, 1]

        # Multi-Layer Pixel-Level Feature Cross-Activation

        fore_mask = (s_y[:, :, 0, :, :] == 1).float().unsqueeze(2)
        masked_img = fore_mask * s_x[:, :, 0, :, :, :] # x_sf
        mask_corrs = []
        for i in range(self.shot):
            with torch.no_grad():
                mask_img = F.interpolate(masked_img[:, i, :, :, :], size=(self.mask_size, self.mask_size),
                                         mode='nearest')
                mask_support_feats, mask_support_backbone_layers = self.extract_feats(mask_img,
                                                                                      [self.layer0, self.layer1,
                                                                                       self.layer2,
                                                                                       self.layer3, self.layer4],
                                                                                      self.feat_ids,
                                                                                      sum(self.nsimlairy),
                                                                                      self.bottleneck_ids,
                                                                                      self.lids)
            mask_corr = FCA(query_feats, mask_support_feats, self.serial, self.corr_blocks,
                                            self.stack_ids)
            mask_corrs.append(mask_corr)

        mask_corrs_shot = list(map(list, zip(*mask_corrs)))
        mask_hyper_4 = torch.stack(mask_corrs_shot[0], 1)
        mask_hyper_4 = (weight_soft.unsqueeze(dim=-1) * mask_hyper_4).sum(1)
        mask_hyper_3 = torch.stack(mask_corrs_shot[1], 1)
        mask_hyper_3 = (weight_soft.unsqueeze(dim=-1) * mask_hyper_3).sum(1)
        if self.vgg:
            mask_hyper_2 = torch.stack(mask_corrs_shot[2], 1)
            mask_hyper_2 = (weight_soft.unsqueeze(dim=-1) * mask_hyper_2).sum(1)
            mask_hyper_2 = F.interpolate(mask_hyper_2,
                                         size=(mask_hyper_3.size(2), mask_hyper_3.size(3)), mode='bilinear',
                                         align_corners=True)
        else:
            mask_hyper_2 = torch.stack(mask_corrs_shot[2], 1)
            mask_hyper_2 = (weight_soft.unsqueeze(dim=-1) * mask_hyper_2).sum(1)
        mask_hyper_final = torch.cat([mask_hyper_2, mask_hyper_3, mask_hyper_4], 1)

        # Prior Mask Generation
        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
            q = query_backbone_layers[4]
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.reshape(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

            tmp_supp = s
            tmp_supp = tmp_supp.reshape(bsize, ch_sz, -1)
            tmp_supp = tmp_supp.permute(0, 2, 1)
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True)

            similarity = torch.bmm(tmp_supp, tmp_query) / (torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)
            similarity = similarity.max(1)[0].reshape(bsize, sp_sz * sp_sz)
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1)) / (
                    similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.reshape(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query,
                                       size=(query_backbone_layers[3].size()[2], query_backbone_layers[3].size()[3]),
                                       mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)
        corr_query_mask = torch.cat(corr_query_mask_list, 1)
        corr_query_mask = (weight_soft * corr_query_mask).sum(1, True)

        # low-level Prototype
        supp_pro = torch.stack(supp_pro_list, 1)  # [bs, 256, shot, 1]
        supp_pro = (weight_soft.unsqueeze(dim=-1) * supp_pro).sum(1) # P_p

        # high-level prototype generation
        query_final_clone = query_feat.clone()
        supp_map_pro = torch.cat(supp_map_pro_list, 2)  # [bs, 256, shot, 1]
        supp_map_pro = (weight_soft.permute(0, 2, 1, 3) * supp_map_pro).sum(2, True)
        query_pro_list = []
        posterior_mask_list = []
        for i in range(self.shot):
            sim_mask = corr_query_mask_list[i].clone() # m_sim_g
            sim_mask = torch.where(sim_mask > 0.8, 1, 0).float() #m_b_g
            query_list = [query_backbone_layers[4]]
            posterior_mask = prior_mask_optimization(query_list, sim_mask)
            posterior_mask_list.append(posterior_mask)
            post_mask = posterior_mask.clone() # m_sim_o
            post_mask = torch.where(post_mask > 0.8, 1, 0).float() # m_b_o
            query_pro = Weighted_GAP(query_final_clone, post_mask) # the image-level query prototype vector
            query_pro_list.append(query_pro)
        final_query_pro = torch.cat(query_pro_list, 2)  # [bs, 256, shot, 1]
        final_query_pro = (weight_soft.permute(0, 2, 1, 3) * final_query_pro).sum(2, True)
        concat_pro = self.alpha[0] * supp_map_pro + self.alpha[1] * final_query_pro # final image-level prototype
        final_posterior_mask = torch.cat(posterior_mask_list, 1)
        final_posterior_mask = (weight_soft * final_posterior_mask).sum(1, True)
        # Tile & Cat
        concat_feat = concat_pro.expand_as(query_feat)
        merge_feat = torch.cat([query_feat, supp_pro, concat_feat, corr_query_mask, final_posterior_mask, mask_hyper_final], 1)  # 256+256+1
        merge_feat = self.init_merge(merge_feat)
        # Base and Meta
        base_out = self.learner_base(query_backbone_layers[4])

        query_meta = self.ASPP_meta(merge_feat)
        query_meta = self.res1_meta(query_meta)  # 1080->256
        query_meta = self.res2_meta(query_meta) + query_meta
        meta_out = self.cls_meta(query_meta)

        meta_out_soft = meta_out.softmax(1)
        base_out_soft = base_out.softmax(1)

        # Classifier Ensemble
        meta_map_bg = meta_out_soft[:, 0:1, :, :]  # [bs, 1, 60, 60]
        meta_map_fg = meta_out_soft[:, 1:, :, :]  # [bs, 1, 60, 60]
        if self.training and self.cls_type == 'Base':
            c_id_array = torch.arange(self.base_classes + 1, device='cuda')
            base_map_list = []
            for b_id in range(bs):
                c_id = cat_idx[0][b_id] + 1
                c_mask = (c_id_array != 0) & (c_id_array != c_id)
                base_map_list.append(base_out_soft[b_id, c_mask, :, :].unsqueeze(0).sum(1, True))
            base_map = torch.cat(base_map_list, 0)

        else:
            base_map = base_out_soft[:, 1:, :, :].sum(1, True)

        est_map = est_val.expand_as(meta_map_fg)

        meta_map_bg = self.gram_merge(torch.cat([meta_map_bg, est_map], dim=1))
        meta_map_fg = self.gram_merge(torch.cat([meta_map_fg, est_map], dim=1))

        merge_map = torch.cat([meta_map_bg, base_map], 1)
        merge_bg = self.cls_merge(merge_map)  # [bs, 1, 60, 60]

        final_out = torch.cat([merge_bg, meta_map_fg], dim=1)

        # Output Part
        if self.zoom_factor != 1:
            meta_out = F.interpolate(meta_out, size=(h, w), mode='bilinear', align_corners=True)
            base_out = F.interpolate(base_out, size=(h, w), mode='bilinear', align_corners=True)
            final_out = F.interpolate(final_out, size=(h, w), mode='bilinear', align_corners=True)

        # Loss
        if self.training:
            main_loss = self.criterion(final_out, y_m.long())
            aux_loss1 = self.criterion(meta_out, y_m.long())
            aux_loss2 = self.criterion(base_out, y_b.long())
            return final_out.max(1)[1], main_loss, aux_loss1, aux_loss2
        else:
            return final_out, meta_out, base_out

    def mask_feature(self, features, support_mask):  # bchw
        bs = features[0].shape[0]
        initSize = ((features[0].shape[-1]) * 2,) * 2
        support_mask = (support_mask).float()
        support_mask = F.interpolate(support_mask, initSize, mode='bilinear', align_corners=True)
        for idx, feature in enumerate(features):
            feat = []
            if support_mask.shape[-1] != feature.shape[-1]:
                support_mask = F.interpolate(support_mask, feature.size()[2:], mode='bilinear', align_corners=True)
            for i in range(bs):
                featI = feature[i].flatten(start_dim=1)  # c,hw
                maskI = support_mask[i].flatten(start_dim=1)  # hw
                featI = featI * maskI
                maskI = maskI.squeeze()
                meanVal = maskI[maskI > 0].mean()
                realSupI = featI[:, maskI >= meanVal]
                if maskI.sum() == 0:
                    realSupI = torch.zeros(featI.shape[0], 1).cuda()
                feat.append(realSupI)  # [b,]ch,w
            features[idx] = feat  # nfeatures ,bs,ch,w
        return features