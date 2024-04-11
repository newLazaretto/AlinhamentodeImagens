import cv2

# Carregando as duas imagens
image1 = cv2.imread('BASE.jpg')
image2 = cv2.imread('reg_image.jpg')

# Definindo a transparência da segunda imagem (alpha)
alpha = 0.5

# Sobrepondo as duas imagens com a função cv2.addWeighted()
overlay = cv2.addWeighted(image1, 0.7, image2, 0.3, 0)

# Exibindo a imagem sobreposta
cv2.imshow('Overlay', overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()