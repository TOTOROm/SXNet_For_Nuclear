# elif function == 3:
#     # from utils.add_noise import *
#     # img = cv2.imread('/workspace/pic/1.jpg')
#     # n1 = add_impulse_noise(img, 50)
#     # n2 = add_text_noise(img, 50)
#     # n3 = add_gaussian_noise(img, 50)
#     # cv2.imwrite('/workspace/pic/n1.jpg', n1)
#     # cv2.imwrite('/workspace/pic/n2.jpg', n2)
#     # cv2.imwrite('/workspace/pic/n3.jpg', n3)
#     for i in Path('/workspace/pic').rglob('*.*'):
#         img = cv2.imread(str(i))
#         crop = img[0:300,0:300,:]
#         cv2.imwrite(str(i)+'.jpg', crop)