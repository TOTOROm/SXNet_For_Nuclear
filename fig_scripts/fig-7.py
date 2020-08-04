# def training_curves(logfile_path):
#     with open(logfile_path) as f:
#         l1 = json.load(f)
#
#     logfile_path2 = logfile_path.replace('l1', 'mse')
#     with open(logfile_path2) as f2:
#         l2 = json.load(f2)
#
#     epochs = [i for i in range(50)]
#     losses_l1 = [i['loss'] * 1e4 for i in l1]
#     psnrs_l1 = [i['psnr'] for i in l1]
#     ssims_l1 = [i['ssim'] for i in l1]
#
#     psnrs_l2 = [i['psnr'] for i in l2]
#     ssims_l2 = [i['ssim'] for i in l2]
#     losses_l2 = [i['loss'] * 1e5 for i in l2]
#
#     def show_psnr():
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#
#         ax.plot(epochs, psnrs_l1, '-', label='L1')
#         ax.plot(epochs, psnrs_l2, '-r', label='L2')
#
#         ax.legend(loc=0)
#         ax.grid()
#         ax.set_xlabel("Epoch")
#         ax.set_ylabel("PSNR (dB)")
#         ax.set_ylim(25, 35)
#
#         plt.savefig("psnr.png")
#         plt.show()
#
#     def show_ssim():
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#
#         ax.plot(epochs, ssims_l1, '-', label='L1')
#         ax.plot(epochs, ssims_l2, '-r', label='L2')
#
#         ax.legend(loc=0)
#         ax.grid()
#         ax.set_xlabel("Epoch")
#         ax.set_ylabel("SSIM")
#         ax.set_ylim(0.7, 1)
#
#         plt.savefig("ssim.png")
#         plt.show()
#
#     def show_loss():
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         ax.set_xlabel("Epoch")
#
#         lns1 = ax.plot(epochs, losses_l1, '-', label='L1 loss')
#
#         ax.set_ylim(130, 380)
#         plt.yticks([])
#         ax2 = ax.twinx()
#         lns2 = ax2.plot(epochs, losses_l2, '-r', label='L2 loss')
#
#         ax2.set_ylim(45, 200)
#
#         lns = lns1 + lns2
#         labs = [l.get_label() for l in lns]
#         ax.legend(lns, labs, loc=0)
#
#         plt.yticks([])
#
#         plt.savefig("loss.png")
#         plt.show()
#
#     show_psnr()
#     show_ssim()
#     show_loss()