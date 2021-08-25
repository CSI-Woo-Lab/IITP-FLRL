import matplotlib.pyplot as plt


metrics_centralized = {'mean_reward': [(0, -2.482360986904242), (1, -0.2197513837300767), (2, -0.09092405678924308), (3, -0.7832408641919858), (4, -0.574054497607658), (5, -0.07517033840067713), (6, -0.06564505750537819), (7, 0.06503736826254683), (8, 0.17932566809500883), (9, -0.017164451335360098), (10, -0.13390069668763316), (11, 0.1081414378774806), (12, -0.35600432710277286), (13, -0.8040694516836151), (14, -1.2369664647153535), (15, -0.09615028051225233), (16, -0.24548443293609584), (17, -0.33485140345849795), (18, -1.0117430087038648), (19, -0.25143136646263886), (20, -0.9546898587599667), (21, -0.678895720187304), (22, -0.30742044296449034), (23, -0.7061038761086831), (24, -0.4803265657358338), (25, -0.6213560719899169), (26, -0.6907263655693503), (27, -1.231023634242033), (28, -0.9816544428825067), (29, -1.0422593816440997), (30, -0.2921682941236349), (31, -2.0925724204531297), (32, -0.527016053831715), (33, -2.494082544354278), (34, 0.17571604873868296), (35, -0.25199728036271607), (36, -1.0546279995217311), (37, -0.03613456260198407), (38, 0.4535572684566057), (39, -0.2213059133840762), (40, -0.20213978889212963)], 'adapt_reward_1': [(0, -2.3944393343799906), (1, -0.2786771605911667), (2, -0.0626204177844997), (3, -0.7930648849319619), (4, -0.47723113628351027), (5, -0.15055868430398958), (6, -0.04507175810041536), (7, -0.010783688320336715), (8, 0.13015715246269566), (9, -0.49844482308192595), (10, -0.09415252645463854), (11, 0.08896301538597524), (12, -0.6316601342066209), (13, -0.3525872590132258), (14, -1.4880543630212653), (15, -0.18714574725693867), (16, -0.08749437518782999), (17, -0.3725223491950464), (18, -0.7088739944212741), (19, -0.1744212036832005), (20, -1.1037939537098054), (21, -0.9290511834296558), (22, -0.2207268843781116), (23, -0.6379910175427255), (24, -0.5148611602013989), (25, -0.5111645740106755), (26, -0.5967605118101733), (27, -1.2882855374271367), (28, -0.9644675741056494), (29, -1.1099973958934015), (30, -0.5199765583827594), (31, -1.9412918310298855), (32, 0.029654940225227605), (33, -2.202810282558121), (34, -0.021685633221312617), (35, -0.5383816941142868), (36, -1.0556052921491117), (37, -0.06267852941639604), (38, 0.40824813768224205), (39, 0.05809840219673221), (40, -0.12573257352094025)],'adapt_reward_2': [(0, -2.225520592600042), (1, -0.46775101632069893), (2, -0.088145456383876), (3, -0.6216656855299212), (4, -0.5875326724170463), (5, -0.1019637122617444), (6, 0.008111389903426691), (7, -0.07357728670790607), (8, 0.134053910951844), (9, -0.31251472409333025), (10, -0.20337333674904134), (11, 0.02930292492183756), (12, -0.5932437407783209), (13, -0.5747824737162488), (14, -1.2145385013366832), (15, -0.15925328002485228), (16, -0.1961413606045017), (17, -0.3704420392133124), (18, -0.8931767398893213), (19, -0.3087795809791396), (20, -0.8026044945058358), (21, -0.6638398073871462), (22, -0.3671423865501856), (23, -0.8839650235364945), (24, -0.5886263654370258), (25, -0.6919335037953056), (26, -0.5824573235374282), (27, -0.9900895199730838), (28, -1.0476452128722031), (29, -0.952829340739547), (30, -0.40723178585381786),(31, -1.6291070734767006), (32, -0.2207210646889138), (33, -2.8543402114246215), (34, 0.20461960044132413), (35, -0.5288314970503724), (36, -0.9367068994057034), (37, -0.06480307533909466), (38, 0.38538016100456585), (39, -0.043418687541815465), (40, -0.2054559323873093)], 'adapt_reward_3': [(0, -1.8961688811681412), (1, -0.32081618262718165), (2, -0.09532877541979529), (3, -0.7395953804038959), (4, -0.2923094397009027), (5, -0.007478917929046772), (6, 0.043973137358186726), (7, 0.003860538070498909), (8, 0.08227264300707902), (9, -0.19727072342472185), (10, -0.1700403915927166), (11, 0.1351998438563578), (12, -0.6364128478442126), (13, -0.44688429810652), (14, -1.8397799946708249), (15, -0.020635260578997082), (16, -0.19508525529175696), (17, -0.3791592430367485), (18, -0.9070123619620676), (19, -0.35227323897052887), (20, -0.8390961803893378), (21, -1.2300045367384858), (22, -0.14372732138904726), (23, -0.7536832401585707), (24, -0.4789948888751144), (25, -0.44784032408184427), (26, -0.7537734620948641), (27, -0.7363597441739943), (28, -0.6534354354759091), (29, -0.44059034156851795), (30, -0.179965144264238), (31, -1.7275999190280753), (32, -0.7614813036687458), (33, -2.3416069380174154), (34, -0.13569059931316133), (35, -0.15199252948094907), (36,-1.0691108200058919), (37, -0.009083000940569491), (38, 0.26332678282453004), (39, -0.11221872999567267), (40, -0.3109810146452912)]}

for key in metrics_centralized:
    li = []
    for i in metrics_centralized[key]:
        li.append(i[1])
    plt.plot(li, label=key)

plt.ylabel("score")
plt.xlabel("rounds")
plt.legend()
plt.grid(True)
# plt.show()
plt.ylim(-4, 6)
plt.savefig("fig.png")