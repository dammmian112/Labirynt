import cv2
import numpy as np
import pygame
import time

# Kolor do rozpoznania
LOWER_COLOR = np.array([20, 100, 100])
UPPER_COLOR = np.array([30, 255, 255])

# Wymiary okna gry
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 500

# Definicja labiryntu
walls = [
    [100, 0, 100, 400],
    [0, 100, 60, 100],
    [30, 200, 250, 200],
    [0, 250, 60, 250],
    [50, 300, 50, 400],
    [50, 400, 100, 400],
    [175, 250, 175, 450],
    [250, 200, 250, 400],
    [0, 450, 400, 450],
    [300, 150, 300, 450],
    [160, 150, 300, 150],
    [100, 100, 350, 100],
    [350, 100, 350, 400],
    [400, 50, 400, 450],
    [130, 50, 400, 50],
    [200, 0, 200, 20],
    [300, 20, 300, 50],
    [350, 25, 450, 25],
    [450, 25, 450, 450],
    [400, 150, 420, 150],
    [480, 150, 500, 150],
    [430, 250, 470, 250],
    [400, 350, 420, 350],
    [480, 350, 500, 350],
    [400, 475, 470, 475],
    [400, 450, 400, 475],
]

# Funkcja rysująca ściany labiryntu
def draw_walls(screen, walls):
    for wall in walls:
        pygame.draw.line(screen, (255, 255, 255), (wall[0], wall[1]), (wall[2], wall[3]), 3)

# Funkcja sprawdzająca kolizję z ścianami
def check_collision(player_pos, player_radius, walls):
    for wall in walls:
        x1, y1, x2, y2 = wall
        if x1 == x2:  # Pionowa ściana
            if abs(player_pos[0] - x1) < player_radius and y1 <= player_pos[1] <= y2:
                return True
        elif y1 == y2:  # Pozioma ściana
            if abs(player_pos[1] - y1) < player_radius and x1 <= player_pos[0] <= x2:
                return True
    return False

# Rozpoczęcie Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Labirynt")
clock = pygame.time.Clock()

# Obiekt do sterowania
player_pos = [50, 50]
player_radius = 10

# Meta
goal_pos = [25, SCREEN_HEIGHT - 25]
goal_radius = 15

# Start kamery
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Nie udało się otworzyć kamery.")
    exit()

# Miernik czasu
start_time = time.time()

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        print("Błąd odczytu z kamery.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        if radius > 10:  # Ignorowanie małych obiektów
            center_x, center_y = int(x), int(y)

            # Określanie kierunku ruchu na podstawie pozycji obiektu
            new_pos = player_pos.copy()
            if center_x < frame.shape[1] // 3:
                new_pos[0] += 5  # Prawo
            elif center_x > 2 * frame.shape[1] // 3:
                new_pos[0] -= 5  # Lewo

            if center_y < frame.shape[0] // 3:
                new_pos[1] -= 5  # Góra
            elif center_y > frame.shape[0] // 3:
                new_pos[1] += 5  # Dół

            # Sprawdzanie kolizji przed zatwierdzeniem ruchu
            if not check_collision(new_pos, player_radius, walls):
                player_pos = new_pos

    # Zabezpieczenie przed wyjściem poza ekran
    player_pos[0] = max(player_radius, min(SCREEN_WIDTH - player_radius, player_pos[0]))
    player_pos[1] = max(player_radius, min(SCREEN_HEIGHT - player_radius, player_pos[1]))

    # Sprawdzenie osiągnięcia mety
    if np.linalg.norm(np.array(player_pos) - np.array(goal_pos)) < player_radius + goal_radius:
        print("Meta osiągnięta!")
        running = False

    # Obsługa zdarzeń Pygame
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Rysowanie w oknie gry
    screen.fill((0, 0, 0))
    draw_walls(screen, walls)
    pygame.draw.circle(screen, (255, 0, 0), player_pos, player_radius)  # Gracz
    pygame.draw.circle(screen, (0, 255, 0), goal_pos, goal_radius)  # Meta

    # Wyświetlanie miernika czasu
    elapsed_time = time.time() - start_time
    font = pygame.font.Font(None, 36)
    timer_text = font.render(f"Czas: {elapsed_time:.2f}s", True, (255, 255, 255))
    screen.blit(timer_text, (10, 10))

    pygame.display.flip()
    clock.tick(30)

# Zakończenie
cap.release()
cv2.destroyAllWindows()
pygame.quit()