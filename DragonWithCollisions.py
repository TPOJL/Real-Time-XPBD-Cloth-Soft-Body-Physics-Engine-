import ctypes
import math
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import numpy as np
import pygame
from pygame.locals import (
    DOUBLEBUF, OPENGL, RESIZABLE, QUIT, KEYDOWN,
    MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, MOUSEWHEEL, VIDEORESIZE,
    K_SPACE, K_r, K_s, K_t, K_c, K_b, K_UP, K_DOWN, K_ESCAPE
)
from OpenGL.GL import (
    glViewport, glClearColor, glClear, glEnable, glDisable,
    glLightfv, glMaterialfv, glColorMaterial, glMatrixMode, glLoadMatrixf,
    glColor3f, glLineWidth, glBegin, glEnd, glVertex3f,
    glEnableClientState, glDisableClientState, glVertexPointer, glNormalPointer,
    glDrawElements, glPolygonOffset, glGetIntegerv, glPushMatrix, glPopMatrix, glTranslatef,
    GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_LIGHTING,
    GL_LIGHT0, GL_LIGHT1, GL_COLOR_MATERIAL, GL_NORMALIZE,
    GL_POSITION, GL_DIFFUSE, GL_AMBIENT, GL_SPECULAR, GL_FRONT_AND_BACK,
    GL_SHININESS, GL_AMBIENT_AND_DIFFUSE, GL_PROJECTION, GL_MODELVIEW,
    GL_LINES, GL_VERTEX_ARRAY, GL_NORMAL_ARRAY, GL_FLOAT, GL_TRIANGLES,
    GL_UNSIGNED_INT, GL_POLYGON_OFFSET_FILL, GL_VIEWPORT
)
from numba import njit


# ==============================================================================
# CZĘŚĆ 1: WCZYTYWANIE SIATEK (TETRAEDRYCZNEJ I WIZUALNEJ) Z PLIKU HTML
# W metodzie Matthiasa Müllera ("Surface Embedding") symulacja fizyczna działa
# na rzadkiej siatce czworościanów (tetMesh), natomiast gęsta siatka trójkątów
# (visMesh, ~60 tys. trójkątów) jest w niej zanurzona za pomocą współrzędnych
# barycentrycznych.
# ==============================================================================

def _extract_array(block: str, name: str, dtype) -> np.ndarray:
    m = re.search(rf"\b{name}\s*:\s*\[([^\]]+)\]", block, re.DOTALL)
    if not m:
        raise ValueError(f"Brak pola '{name}' w definicji siatki HTML.")
    raw = m.group(1).replace("\n", " ").replace("\r", " ").strip().rstrip(",")
    return np.fromstring(raw, sep=",", dtype=dtype)


def load_dragon_meshes(html_filename="12-softBodySkinning.html"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, html_filename)

    if not os.path.isfile(html_path):
        raise FileNotFoundError(f"Nie znaleziono pliku z danymi siatki: {html_path}")

    with open(html_path, "r", encoding="utf-8") as f:
        text = f.read()

    m_tet = re.search(r"var\s+dragonTetMesh\s*=\s*\{", text)
    m_vis = re.search(r"var\s+dragonVisMesh\s*=\s*\{", text)
    if not m_tet or not m_vis:
        raise ValueError("W pliku HTML nie znaleziono obiektów dragonTetMesh / dragonVisMesh.")

    tet_start, vis_start = m_tet.start(), m_vis.start()
    if tet_start < vis_start:
        tet_block, vis_block = text[tet_start:vis_start], text[vis_start:]
    else:
        vis_block, tet_block = text[vis_start:tet_start], text[tet_start:]

    tet_mesh = {
        "verts": _extract_array(tet_block, "verts", np.float32).reshape(-1, 3),
        "tetIds": _extract_array(tet_block, "tetIds", np.int32).reshape(-1, 4),
        "edgeIds": _extract_array(tet_block, "edgeIds", np.int32).reshape(-1, 2),
    }
    vis_mesh = {
        "verts": _extract_array(vis_block, "verts", np.float32).reshape(-1, 3),
        "triIds": _extract_array(vis_block, "triIds", np.int32).reshape(-1, 3),
    }
    return tet_mesh, vis_mesh

# ==============================================================================
# CZĘŚĆ 2: HASZOWANIE PRZESTRZENNE (SPATIAL HASHING) I SKINNING BARYCENTRYCZNY (kazdy wierzolek na wieksza siatke)
# ==============================================================================


#hash
@njit(cache=True)
def hash_coords(xi: int, yi: int, zi: int, table_size: int) -> int:
    h = (np.int64(xi) * 92837111) ^ (np.int64(yi) * 689287499) ^ (np.int64(zi) * 283923481)
    return int(abs(h) % table_size)


@njit(cache=True)
def compute_skinning_info(pos: np.ndarray, tet_ids: np.ndarray, vis_verts: np.ndarray):
    """
    Oblicza wagi barycentryczne (b0, b1, b2, b3) dla każdego wierzchołka siatki
    wizualnej względem otaczającego (lub najbliższego) czworościanu:
        v - p3 = P * [b0, b1, b2]^T,  gdzie P = [p0 - p3 | p1 - p3 | p2 - p3],
        b3 = 1 - b0 - b1 - b2.
    """
    num_vis = vis_verts.shape[0]
    num_tets = tet_ids.shape[0]
    spacing = 0.05
    inv_spacing = 1.0 / spacing
    table_size = 2 * num_vis

    # Budowa siatki haszującej dla wierzchołków wizualnych + (tablice zliczeń prefiksowych)
    cell_start = np.zeros(table_size + 1, dtype=np.int32)
    cell_entries = np.zeros(num_vis, dtype=np.int32)

    for i in range(num_vis):
        xi = int(math.floor(vis_verts[i, 0] * inv_spacing))
        yi = int(math.floor(vis_verts[i, 1] * inv_spacing))
        zi = int(math.floor(vis_verts[i, 2] * inv_spacing))
        cell_start[hash_coords(xi, yi, zi, table_size)] += 1

    start = 0
    for i in range(table_size):
        start += cell_start[i]
        cell_start[i] = start
    cell_start[table_size] = start

    for i in range(num_vis):
        xi = int(math.floor(vis_verts[i, 0] * inv_spacing))
        yi = int(math.floor(vis_verts[i, 1] * inv_spacing))
        zi = int(math.floor(vis_verts[i, 2] * inv_spacing))
        h = hash_coords(xi, yi, zi, table_size)
        cell_start[h] -= 1
        cell_entries[cell_start[h]] = i

    skin_tet = np.full(num_vis, -1, dtype=np.int32)
    skin_bary = np.zeros((num_vis, 3), dtype=np.float32)
    min_dist = np.full(num_vis, 1e30, dtype=np.float32)
    border = 0.05


    # Dla każdego czworościanu wyznaczamy kulę otaczającą i odpytujemy Spatial Hash (optymizacja zeby nie sprawdzac wszystkie 60000 wierzolkow)
    for i in range(num_tets):
        id0, id1, id2, id3 = tet_ids[i, 0], tet_ids[i, 1], tet_ids[i, 2], tet_ids[i, 3]

        cx = 0.25 * (pos[id0, 0] + pos[id1, 0] + pos[id2, 0] + pos[id3, 0])
        cy = 0.25 * (pos[id0, 1] + pos[id1, 1] + pos[id2, 1] + pos[id3, 1])
        cz = 0.25 * (pos[id0, 2] + pos[id1, 2] + pos[id2, 2] + pos[id3, 2])

        r_max = 0.0
        for j in range(4):
            pid = tet_ids[i, j]
            r = math.sqrt((cx - pos[pid, 0]) ** 2 + (cy - pos[pid, 1]) ** 2 + (cz - pos[pid, 2]) ** 2)
            if r > r_max:
                r_max = r
        r_max += border
        r_max2 = r_max * r_max

        # Odwracanie macierzy 3x3 krawędzi czworościanu: P^-1 dla barycentrycznych wspol.
        c0x, c0y, c0z = pos[id0, 0] - pos[id3, 0], pos[id0, 1] - pos[id3, 1], pos[id0, 2] - pos[id3, 2]
        c1x, c1y, c1z = pos[id1, 0] - pos[id3, 0], pos[id1, 1] - pos[id3, 1], pos[id1, 2] - pos[id3, 2]
        c2x, c2y, c2z = pos[id2, 0] - pos[id3, 0], pos[id2, 1] - pos[id3, 1], pos[id2, 2] - pos[id3, 2]

        r0x, r0y, r0z = c1y * c2z - c1z * c2y, c1z * c2x - c1x * c2z, c1x * c2y - c1y * c2x
        det = c0x * r0x + c0y * r0y + c0z * r0z
        if det == 0.0:
            continue
        inv_det = 1.0 / det

        r0x, r0y, r0z = r0x * inv_det, r0y * inv_det, r0z * inv_det
        r1x = (c2y * c0z - c2z * c0y) * inv_det
        r1y = (c2z * c0x - c2x * c0z) * inv_det
        r1z = (c2x * c0y - c2y * c0x) * inv_det
        r2x = (c0y * c1z - c0z * c1y) * inv_det
        r2y = (c0z * c1x - c0x * c1z) * inv_det
        r2z = (c0x * c1y - c0y * c1x) * inv_det

        #tu sprawdzamy wierzolki wlasnie w tej kulie kto obok (juz nizej to faktycznie cube)
        x0, x1 = int(math.floor((cx - r_max) * inv_spacing)), int(math.floor((cx + r_max) * inv_spacing))
        y0, y1 = int(math.floor((cy - r_max) * inv_spacing)), int(math.floor((cy + r_max) * inv_spacing))
        z0, z1 = int(math.floor((cz - r_max) * inv_spacing)), int(math.floor((cz + r_max) * inv_spacing))

        for xi in range(x0, x1 + 1):
            for yi in range(y0, y1 + 1):
                for zi in range(z0, z1 + 1):
                    h = hash_coords(xi, yi, zi, table_size)
                    for k in range(cell_start[h], cell_start[h + 1]):
                        vid = cell_entries[k]
                        # Jeśli znaleźliśmy już czworościan zawierający ten punkt (dist <= 0), pomijamy
                        if min_dist[vid] <= 0.0:
                            continue

                        vx, vy, vz = vis_verts[vid, 0], vis_verts[vid, 1], vis_verts[vid, 2]
                        if (vx - cx) ** 2 + (vy - cy) ** 2 + (vz - cz) ** 2 > r_max2:
                            continue

                        px, py, pz = vx - pos[id3, 0], vy - pos[id3, 1], vz - pos[id3, 2]
                        b0 = r0x * px + r0y * py + r0z * pz
                        b1 = r1x * px + r1y * py + r1z * pz
                        b2 = r2x * px + r2y * py + r2z * pz
                        b3 = 1.0 - b0 - b1 - b2

                        # Odległość barycentryczna: punkt leży wewnątrz wtw, gdy wszystkie b_k >= 0 (jak w filmiku)
                        dist = max(0.0, -b0, -b1, -b2, -b3)
                        if dist < min_dist[vid]:
                            min_dist[vid] = dist
                            skin_tet[vid] = i
                            skin_bary[vid, 0] = b0
                            skin_bary[vid, 1] = b1
                            skin_bary[vid, 2] = b2

    return skin_tet, skin_bary


# ==============================================================================
# >>>>>>>>>>>>>>>>>>>>>> POCZĄTEK RDZENIA METODY XPBD <<<<<<<<<<<<<<<<<<<<<<<<<<
# Extended Position Based Dynamics (Macklin, Müller, Chentanez - 2016)
#
# Cały jeden podkrok czasowy (substep o długości sdt = dt / numSubsteps)
# składa się dokładnie z 3 faz:
#   KROK 1 (preSolve):   Jawne całkowanie sił zewnętrznych (grawitacja) -> predykcja x
#   KROK 2 (solve):      Projekcja więzów (Gauss-Seidel):
#                          2A - Więzy długości krawędzi (sprężystość kształtu)
#                          2B - Więzy objętości czworościanów (zachowanie objętości)
#                          2C - Więzy kolizji (wewnętrzne + między ciałami + z kulą)
#   KROK 3 (postSolve):  Wyliczenie nowych prędkości v = (x - x_prev) / sdt
# ==============================================================================

@njit(cache=True)
def xpbd_step_1_pre_solve(pos, prev_pos, vel, inv_mass, radii, gravity, sdt, ground_friction):
    """
    XPBD - KROK 1 (preSolve):
    Całkowanie symplektycznym schematem Eulera oraz obsługa kontaktu z podłożem y = 0.
    Dla każdej cząstki o masie skończonej (inv_mass > 0):
        1. v_i <- v_i + sdt * g
        2. x_prev_i <- x_i          (zapamiętujemy pozycję sprzed korekt więzów!)
        3. x_i <- x_i + sdt * v_i   (pozycja przewidywana)
    """
    for i in range(pos.shape[0]):
        if inv_mass[i] == 0.0:  # inv_mass == 0 oznacza cząstkę zablokowaną (np. złapaną myszką)
            prev_pos[i, 0] = pos[i, 0]
            prev_pos[i, 1] = pos[i, 1]
            prev_pos[i, 2] = pos[i, 2]
            continue

        vel[i, 0] += gravity[0] * sdt
        vel[i, 1] += gravity[1] * sdt
        vel[i, 2] += gravity[2] * sdt

        prev_pos[i, 0] = pos[i, 0]
        prev_pos[i, 1] = pos[i, 1]
        prev_pos[i, 2] = pos[i, 2]

        pos[i, 0] += vel[i, 0] * sdt
        pos[i, 1] += vel[i, 1] * sdt
        pos[i, 2] += vel[i, 2] * sdt

        # Kolizja z podłogą y = 0 z uwzględnieniem promienia cząstki oraz tarcia
        min_y = 0.5 * radii[i]
        if pos[i, 1] < min_y:
            # Tarcie poziome proporcjonalne do współczynnika ground_friction
            pos[i, 0] = pos[i, 0] - ground_friction * (pos[i, 0] - prev_pos[i, 0])
            pos[i, 2] = pos[i, 2] - ground_friction * (pos[i, 2] - prev_pos[i, 2])
            pos[i, 1] = min_y

        # Niewidzialne ściany boczne areny [-3.5, 3.5], żeby smoki nie uciekały poza planszę
        wall = 3.5
        if pos[i, 0] < -wall:
            pos[i, 0] = -wall
        elif pos[i, 0] > wall:
            pos[i, 0] = wall
        if pos[i, 2] < -wall:
            pos[i, 2] = -wall
        elif pos[i, 2] > wall:
            pos[i, 2] = wall


@njit(cache=True)
def xpbd_step_2a_solve_edges(pos, inv_mass, edge_ids, edge_lengths, compliance, sdt):
    """
    XPBD - KROK 2A (Więzy odległości / krawędzi):
    Funkcja więzu dla krawędzi (i, j) o długości spoczynkowej l_0:
        C(x_i, x_j) = |x_i - x_j| - l_0
    Gradient więzu to wektor jednostkowy wzdłuż krawędzi: n = (x_i - x_j) / |x_i - x_j|.
    Kluczowa formuła XPBD z podatnością alpha_tilde = compliance / sdt^2:
        s = -C / (w_i + w_j + alpha_tilde)
        dx_i = +s * w_i * n
        dx_j = -s * w_j * n
    Dzięki członowi alpha_tilde / sdt^2 sztywność materiału NIE zależy od liczby podkroków!
    """
    alpha = compliance / (sdt * sdt) #compilance prezyny
    for i in range(edge_ids.shape[0]):
        #nasze dwa wierz.
        id0 = edge_ids[i, 0]
        id1 = edge_ids[i, 1]
        w0 = inv_mass[id0]
        w1 = inv_mass[id1]
        w = w0 + w1
        if w == 0.0:
            continue

        #dlugosc + C
        dx = pos[id0, 0] - pos[id1, 0]
        dy = pos[id0, 1] - pos[id1, 1]
        dz = pos[id0, 2] - pos[id1, 2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length == 0.0:
            continue

        inv_len = 1.0 / length
        gx, gy, gz = dx * inv_len, dy * inv_len, dz * inv_len #wektor poprawki

        C = length - edge_lengths[i] #na ile rozciegneta
        s = -C / (w + alpha) #na ile musimy przesunac punkty zeby wrucic nasze dlugosci krawedzi

        #wlasnie ta poprawka pozycji zeby wszystko wyrownac
        pos[id0, 0] += gx * (s * w0)
        pos[id0, 1] += gy * (s * w0)
        pos[id0, 2] += gz * (s * w0)
        pos[id1, 0] -= gx * (s * w1)
        pos[id1, 1] -= gy * (s * w1)
        pos[id1, 2] -= gz * (s * w1)


# jak poprzednie ale na czworosciany (bo byli wtedy puste i mogliby deformowac na wewnatrz)
@njit(cache=True)
def xpbd_step_2b_solve_volumes(pos, inv_mass, tet_ids, rest_vol, vol_id_order, compliance, sdt):
    """
    XPBD - KROK 2B (Więzy zachowania objętości czworościanu):
    Objętość ze znakiem czworościanu (p0, p1, p2, p3):
        V = 1/6 * ((p1 - p0) x (p2 - p0)) . (p3 - p0)
    Funkcja więzu:
        C(p0, p1, p2, p3) = V - V_rest
    Gradient więzu względem wierzchołka j (grad_j C) to wektor normalny przeciwległej
    ściany trójkątnej podzielony przez 6 (skierowany na zewnątrz czworościanu).
    Mnożnik Lagrange'a:
        s = -C / (sum_{j=0..3} (w_j * |grad_j|^2) + alpha_tilde)
    Ponieważ V jest objętością ZE ZNAKIEM, po zgnieceniu smoka (Squash, gdy V < 0)
    gradienty automatycznie wypychają odwrócone czworościany z powrotem do pierwotnego kształtu!
    """
    alpha = compliance / (sdt * sdt)
    grads = np.empty((4, 3), dtype=np.float32)

    #tu dla kazdego wierzolku czworosciany licymy w ktora strone i z ktora moca musimy ten wierzolek przesunac zeby wrocic volume do poczatkowego
    for i in range(tet_ids.shape[0]):
        w = 0.0
        for j in range(4):
            id0 = tet_ids[i, vol_id_order[j, 0]]
            id1 = tet_ids[i, vol_id_order[j, 1]]
            id2 = tet_ids[i, vol_id_order[j, 2]]

            e1x, e1y, e1z = pos[id1, 0] - pos[id0, 0], pos[id1, 1] - pos[id0, 1], pos[id1, 2] - pos[id0, 2]
            e2x, e2y, e2z = pos[id2, 0] - pos[id0, 0], pos[id2, 1] - pos[id0, 1], pos[id2, 2] - pos[id0, 2]

            cx = (e1y * e2z - e1z * e2y) / 6.0
            cy = (e1z * e2x - e1x * e2z) / 6.0
            cz = (e1x * e2y - e1y * e2x) / 6.0

            grads[j, 0], grads[j, 1], grads[j, 2] = cx, cy, cz
            w += inv_mass[tet_ids[i, j]] * (cx * cx + cy * cy + cz * cz)

        if w == 0.0:
            continue

        p0, p1, p2, p3 = tet_ids[i, 0], tet_ids[i, 1], tet_ids[i, 2], tet_ids[i, 3]
        d1x, d1y, d1z = pos[p1, 0] - pos[p0, 0], pos[p1, 1] - pos[p0, 1], pos[p1, 2] - pos[p0, 2]
        d2x, d2y, d2z = pos[p2, 0] - pos[p0, 0], pos[p2, 1] - pos[p0, 1], pos[p2, 2] - pos[p0, 2]
        d3x, d3y, d3z = pos[p3, 0] - pos[p0, 0], pos[p3, 1] - pos[p0, 1], pos[p3, 2] - pos[p0, 2]

        vol = ((d1y * d2z - d1z * d2y) * d3x + (d1z * d2x - d1x * d2z) * d3y + (d1x * d2y - d1y * d2x) * d3z) / 6.0
        C = vol - rest_vol[i] #na ile rozciegneta
        s = -C / (w + alpha) #na ile musimy przesunac punkty zeby wrucic nasze dlugosci krawedzi

        for j in range(4):
            pid = tet_ids[i, j]
            sw = s * inv_mass[pid]
            pos[pid, 0] += grads[j, 0] * sw
            pos[pid, 1] += grads[j, 1] * sw
            pos[pid, 2] += grads[j, 2] * sw


@njit(cache=True)
def xpbd_step_2c_solve_collisions(
    pos, prev_pos, rest_pos, inv_mass, radii, body_ids,
    cell_start, cell_entries, spacing, friction,
    sphere_active, sphere_pos, sphere_prev_pos, sphere_radius
):
    """
    XPBD - KROK 2C (ROZSZERZENIE NA 5 PUNKTÓW: Obsługa kolizji w XPBD):
    Obsługuje JEDNOCZEŚNIE trzy rodzaje zderzeń:
      1. Kolizje między różnymi ciałami miękkimi (np. dwa smoki zderzające się w locie).
      2. Kolizje wewnętrzne / autokolizje (ogon, głowa i łapy tego samego smoka nie przenikają się).
      3. Kolizje z zewnętrznym ciałem sztywnym (interaktywna kula sterowana myszką).

    Algorytm:
      A. Budujemy Spatial Hash dla wszystkich aktywnych cząstek w czasie O(N).
      B. Dla każdej cząstki i przeglądamy sąsiedztwo 3x3x3 komórek siatki.
      C. Jeśli odległość d = |x_i - x_j| < r_i + r_j:
         - Dla cząstek TEGO SAMEGO ciała sprawdzamy dodatkowo odległość spoczynkową
           |x_i^rest - x_j^rest| > 1.15 * (r_i + r_j), aby sąsiednie wierzchołki w siatce
           nie odpychały się sztucznie w stanie spoczynku!
         - Rozwiązujemy więz nierównościowy XPBD (alpha = 0, twardy kontakt):
               C = d - (r_i + r_j) < 0
               dx_i = -C * (w_i / (w_i + w_j)) * n
               dx_j = +C * (w_j / (w_i + w_j)) * n
         - Aplikujemy pozycyjne tarcie Coulomba (tłumienie poślizgu stycznego do normalnej n).
    """
    n_particles = pos.shape[0]
    table_size = cell_start.shape[0] - 1
    inv_spacing = 1.0 / spacing

    # --- A. Budowa siatki Spatial Hash ---
    cell_start.fill(0)
    for i in range(n_particles):
        xi = int(math.floor(pos[i, 0] * inv_spacing))
        yi = int(math.floor(pos[i, 1] * inv_spacing))
        zi = int(math.floor(pos[i, 2] * inv_spacing))
        cell_start[hash_coords(xi, yi, zi, table_size)] += 1

    start = 0
    for i in range(table_size):
        start += cell_start[i]
        cell_start[i] = start
    cell_start[table_size] = start

    for i in range(n_particles):
        xi = int(math.floor(pos[i, 0] * inv_spacing))
        yi = int(math.floor(pos[i, 1] * inv_spacing))
        zi = int(math.floor(pos[i, 2] * inv_spacing))
        h = hash_coords(xi, yi, zi, table_size)
        cell_start[h] -= 1
        cell_entries[cell_start[h]] = i

    # --- B. Rozwiązywanie zderzeń cząstka-cząstka (między ciałami oraz wewnętrznych) ---

    #spatial hashing
    for i in range(n_particles):
        wi = inv_mass[i]
        ri = radii[i]
        bi = body_ids[i]

        px, py, pz = pos[i, 0], pos[i, 1], pos[i, 2]
        cxi = int(math.floor(px * inv_spacing))
        cyi = int(math.floor(py * inv_spacing))
        czi = int(math.floor(pz * inv_spacing))

        # Przeszukujemy 27 sąsiednich komórek siatki przestrzennej
        for xi in range(cxi - 1, cxi + 2):
            for yi in range(cyi - 1, cyi + 2):
                for zi in range(czi - 1, czi + 2):
                    h = hash_coords(xi, yi, zi, table_size)
                    for k in range(cell_start[h], cell_start[h + 1]):
                        j = cell_entries[k]
                        # Każdą parę (i, j) rozpatrujemy tylko raz (j > i)
                        if j <= i:
                            continue

                        wj = inv_mass[j]
                        w_sum = wi + wj
                        if w_sum == 0.0:
                            continue

                        min_dist = ri + radii[j]
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dz = pos[i, 2] - pos[j, 2]
                        d2 = dx * dx + dy * dy + dz * dz

                        #sprawdzamy odleglosc dowch punktow, jak ok to "continue"
                        if d2 >= min_dist * min_dist or d2 < 1e-12:
                            continue

                        # Filtr dla kolizji WEWNĘTRZNYCH (w obrębie tego samego smoka):
                        # Ignorujemy pary, które już w pozycji spoczynkowej leżały blisko siebie
                        if bi == body_ids[j]:
                            rdx = rest_pos[i, 0] - rest_pos[j, 0]
                            rdy = rest_pos[i, 1] - rest_pos[j, 1]
                            rdz = rest_pos[i, 2] - rest_pos[j, 2]
                            rest_thresh = 1.15 * min_dist
                            if rdx * rdx + rdy * rdy + rdz * rdz < rest_thresh * rest_thresh: #blokada zeby pomiedzy soba nie odbijali sie punkty tego samego drakonu
                                continue

                        d = math.sqrt(d2)
                        nx, ny, nz = dx / d, dy / d, dz / d  # kierunek korekcji (normalna kontaktu)

                        # Głębokość penetracji (odpowiednik błędu C < 0)
                        pen = min_dist - d  # wielkość błędu (na ile weszlo wewnatrz)
                        corr = pen / w_sum  # współczynnik korekcji (odpowiednik 's' z poprzednich kroków)

                        # Korekta normalna pozycji (rozdzielenie przenikających się cząstek)
                        pos[i, 0] += nx * (corr * wi)
                        pos[i, 1] += ny * (corr * wi)
                        pos[i, 2] += nz * (corr * wi)

                        pos[j, 0] -= nx * (corr * wj)
                        pos[j, 1] -= ny * (corr * wj)
                        pos[j, 2] -= nz * (corr * wj)


                        # Tarcie styczne w punkcie kontaktu (XPBD Position-Based Friction) – żeby wyglądało dobrze, a nie po prostu ślizgało się po mapie
                        rel_dx = (pos[i, 0] - prev_pos[i, 0]) - (pos[j, 0] - prev_pos[j, 0]) #prawdziwe przesuniecie
                        rel_dy = (pos[i, 1] - prev_pos[i, 1]) - (pos[j, 1] - prev_pos[j, 1])
                        rel_dz = (pos[i, 2] - prev_pos[i, 2]) - (pos[j, 2] - prev_pos[j, 2])

                        vn = rel_dx * nx + rel_dy * ny + rel_dz * nz
                        tx = rel_dx - vn * nx
                        ty = rel_dy - vn * ny
                        tz = rel_dz - vn * nz

                        t_len = math.sqrt(tx * tx + ty * ty + tz * tz)
                        if t_len > 1e-8:
                            # Ograniczenie Coulomba: korekta styczna nie przekracza mu * pen
                            f_mag = min(friction * pen, t_len) / (t_len * w_sum)
                            pos[i, 0] -= tx * (f_mag * wi)
                            pos[i, 1] -= ty * (f_mag * wi)
                            pos[i, 2] -= tz * (f_mag * wi)

                            pos[j, 0] += tx * (f_mag * wj)
                            pos[j, 1] += ty * (f_mag * wj)
                            pos[j, 2] += tz * (f_mag * wj)

@njit(cache=True)
def xpbd_step_3_post_solve(pos, prev_pos, vel, inv_mass, sdt):
    """
    XPBD - KROK 3 (postSolve):
    Kluczowy moment każdej metody Position Based Dynamics!
    Prędkości NIE są całkowane z sił sprężystości czy impulsów kolizji — zamiast tego
    wyliczamy je bezpośrednio ze zmiany pozycji wywołanej przez projekcję więzów:
        v_i <- (x_i - x_prev_i) / sdt
    To właśnie gwarantuje bezwarunkową stabilność numeryczną XPBD.
    """
    inv_dt = 1.0 / sdt
    for i in range(pos.shape[0]):
        if inv_mass[i] == 0.0:
            continue
        vel[i, 0] = (pos[i, 0] - prev_pos[i, 0]) * inv_dt
        vel[i, 1] = (pos[i, 1] - prev_pos[i, 1]) * inv_dt
        vel[i, 2] = (pos[i, 2] - prev_pos[i, 2]) * inv_dt

# ==============================================================================
# >>>>>>>>>>>>>>>>>>>>>>> KONIEC RDZENIA METODY XPBD <<<<<<<<<<<<<<<<<<<<<<<<<<<
# ==============================================================================


# ==============================================================================
# CZĘŚĆ 3: AKTUALIZACJA SIATKI WIZUALNEJ (SKINNING) ORAZ RAYCASTING MYSZY
# ==============================================================================

@njit(cache=True)
def update_vis_mesh_kernel(pos, vert_offset, tet_ids, skin_tet, skin_bary, tri_ids, vis_pos, vis_normals):
    """Przelicza pozycje 60 tys. trójkątów wizualnych z deformacji czworościanów i liczy normalne."""
    num_vis = vis_pos.shape[0]
    for i in range(num_vis):
        t = skin_tet[i]
        if t < 0:
            continue
        b0, b1, b2 = skin_bary[i, 0], skin_bary[i, 1], skin_bary[i, 2]
        b3 = 1.0 - b0 - b1 - b2

        # Indeksy globalne 4 wierzchołków czworościanu
        id0 = tet_ids[t, 0] + vert_offset
        id1 = tet_ids[t, 1] + vert_offset
        id2 = tet_ids[t, 2] + vert_offset
        id3 = tet_ids[t, 3] + vert_offset

        vis_pos[i, 0] = b0 * pos[id0, 0] + b1 * pos[id1, 0] + b2 * pos[id2, 0] + b3 * pos[id3, 0]
        vis_pos[i, 1] = b0 * pos[id0, 1] + b1 * pos[id1, 1] + b2 * pos[id2, 1] + b3 * pos[id3, 1]
        vis_pos[i, 2] = b0 * pos[id0, 2] + b1 * pos[id1, 2] + b2 * pos[id2, 2] + b3 * pos[id3, 2]

        vis_normals[i, 0] = 0.0
        vis_normals[i, 1] = 0.0
        vis_normals[i, 2] = 0.0

    for i in range(tri_ids.shape[0]):
        i0, i1, i2 = tri_ids[i, 0], tri_ids[i, 1], tri_ids[i, 2]
        e1x, e1y, e1z = vis_pos[i1, 0] - vis_pos[i0, 0], vis_pos[i1, 1] - vis_pos[i0, 1], vis_pos[i1, 2] - vis_pos[i0, 2]
        e2x, e2y, e2z = vis_pos[i2, 0] - vis_pos[i0, 0], vis_pos[i2, 1] - vis_pos[i0, 1], vis_pos[i2, 2] - vis_pos[i0, 2]

        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x

        vis_normals[i0, 0] += nx
        vis_normals[i0, 1] += ny
        vis_normals[i0, 2] += nz
        vis_normals[i1, 0] += nx
        vis_normals[i1, 1] += ny
        vis_normals[i1, 2] += nz
        vis_normals[i2, 0] += nx
        vis_normals[i2, 1] += ny
        vis_normals[i2, 2] += nz

    for i in range(num_vis):
        nx, ny, nz = vis_normals[i, 0], vis_normals[i, 1], vis_normals[i, 2]
        l2 = nx * nx + ny * ny + nz * nz
        if l2 > 1e-20:
            inv_l = 1.0 / math.sqrt(l2)
            vis_normals[i, 0] = nx * inv_l
            vis_normals[i, 1] = ny * inv_l
            vis_normals[i, 2] = nz * inv_l


@njit(cache=True)
def raycast_triangles(ray_orig: np.ndarray, ray_dir: np.ndarray, vis_pos: np.ndarray, tri_ids: np.ndarray) -> float:
    """Przecięcie promienia myszy z trójkątami (algorytm Möllera-Trumbore'a)."""
    min_t = 1e30
    ox, oy, oz = ray_orig[0], ray_orig[1], ray_orig[2]
    dx, dy, dz = ray_dir[0], ray_dir[1], ray_dir[2]

    for i in range(tri_ids.shape[0]):
        i0, i1, i2 = tri_ids[i, 0], tri_ids[i, 1], tri_ids[i, 2]
        v0x, v0y, v0z = vis_pos[i0, 0], vis_pos[i0, 1], vis_pos[i0, 2]
        e1x, e1y, e1z = vis_pos[i1, 0] - v0x, vis_pos[i1, 1] - v0y, vis_pos[i1, 2] - v0z
        e2x, e2y, e2z = vis_pos[i2, 0] - v0x, vis_pos[i2, 1] - v0y, vis_pos[i2, 2] - v0z

        hx = dy * e2z - dz * e2y
        hy = dz * e2x - dx * e2z
        hz = dx * e2y - dy * e2x

        a = e1x * hx + e1y * hy + e1z * hz
        if -1e-7 < a < 1e-7:
            continue

        f = 1.0 / a
        sx, sy, sz = ox - v0x, oy - v0y, oz - v0z
        u = f * (sx * hx + sy * hy + sz * hz)
        if u < 0.0 or u > 1.0:
            continue

        qx = sy * e1z - sz * e1y
        qy = sz * e1x - sx * e1z
        qz = sx * e1y - sy * e1x

        v = f * (dx * qx + dy * qy + dz * qz)
        if v < 0.0 or u + v > 1.0:
            continue

        t = f * (e2x * qx + e2y * qy + e2z * qz)
        if 1e-4 < t < min_t:
            min_t = t

    return min_t


def raycast_sphere(ray_orig: np.ndarray, ray_dir: np.ndarray, center: np.ndarray, radius: float) -> float:
    oc = ray_orig - center
    b = float(np.dot(oc, ray_dir))
    c = float(np.dot(oc, oc) - radius * radius)
    disc = b * b - c
    if disc < 0.0:
        return 1e30
    t = -b - math.sqrt(disc)
    return t if t > 1e-4 else 1e30


# ==============================================================================
# CZĘŚĆ 4: GLOBALNY ŚWIAT FIZYKI XPBD (XPBDWorld + wizualne instancje smoków)
# Łączymy cząstki wszystkich ciał w jedną wspólną tablicę globalną, aby solwer
# kolizji mógł rozwiązywać zderzenia między dowolnymi ciałami w jednej pętli!
# ==============================================================================

class SoftBodyInstance:
    """Reprezentuje pojedyncze miękkie ciało zanurzone w globalnym solwerze XPBDWorld."""
    def __init__(self, body_id: int, vert_offset: int, num_verts: int,
                 local_tet_ids: np.ndarray, global_edge_ids: np.ndarray,
                 skin_tet: np.ndarray, skin_bary: np.ndarray,
                 vis_verts: np.ndarray, tri_ids: np.ndarray, color: tuple):
        self.body_id = body_id
        self.vert_offset = vert_offset
        self.num_verts = num_verts
        self.local_tet_ids = local_tet_ids
        self.edge_indices_gl = np.ascontiguousarray(global_edge_ids.ravel(), dtype=np.uint32)

        self.skin_tet = skin_tet
        self.skin_bary = skin_bary
        self.tri_ids = tri_ids
        self.tri_indices_gl = np.ascontiguousarray(tri_ids.ravel(), dtype=np.uint32)

        self.vis_pos = vis_verts.copy()
        self.vis_normals = np.zeros_like(self.vis_pos)
        self.color = color


class XPBDWorld:
    """Globalny silnik XPBD zarządzający wszystkimi cząstkami, więzami i kolizjami."""
    VOL_ID_ORDER = np.array([[1, 3, 2], [0, 2, 3], [0, 3, 1], [0, 1, 2]], dtype=np.int32)

    def __init__(self, tet_mesh: dict, vis_mesh: dict):
        # 1. Obliczamy wagi skinningu tylko raz dla bazowego smoka (duża oszczędność czasu startu!)
        base_tet_verts = np.ascontiguousarray(tet_mesh["verts"], dtype=np.float32)
        base_tet_ids = np.ascontiguousarray(tet_mesh["tetIds"], dtype=np.int32)
        base_edge_ids = np.ascontiguousarray(tet_mesh["edgeIds"], dtype=np.int32)
        base_vis_verts = np.ascontiguousarray(vis_mesh["verts"], dtype=np.float32)
        base_tri_ids = np.ascontiguousarray(vis_mesh["triIds"], dtype=np.int32)

        skin_tet, skin_bary = compute_skinning_info(base_tet_verts, base_tet_ids, base_vis_verts)

        self.bodies = []
        all_init_pos = []
        all_rest_pos = []
        all_body_ids = []
        all_tet_ids = []
        all_edge_ids = []

        # Konfiguracja dwóch smoków:
        # Smok 0 (pomarańczowy) na dole, Smok 1 (turkusowy, obrócony o 180 st.) spada na niego z góry!
        configs = [
            {"offset": np.array([-0.10, 0.0, 0.0], dtype=np.float32), "rot_y": 0.0, "color": (0.969, 0.541, 0.114)},
            {"offset": np.array([0.12, 1.35, 0.05], dtype=np.float32), "rot_y": math.pi, "color": (0.180, 0.720, 0.850)},
        ]

        vert_offset = 0
        for b_id, cfg in enumerate(configs):
            c, s = math.cos(cfg["rot_y"]), math.sin(cfg["rot_y"])
            rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)

            v_world = (base_tet_verts @ rot.T) + cfg["offset"]
            n_v = v_world.shape[0]

            g_tets = base_tet_ids + vert_offset
            g_edges = base_edge_ids + vert_offset

            all_init_pos.append(v_world)
            all_rest_pos.append(base_tet_verts.copy())
            all_body_ids.append(np.full(n_v, b_id, dtype=np.int32))
            all_tet_ids.append(g_tets)
            all_edge_ids.append(g_edges)

            inst = SoftBodyInstance(
                body_id=b_id,
                vert_offset=vert_offset,
                num_verts=n_v,
                local_tet_ids=base_tet_ids,
                global_edge_ids=g_edges,
                skin_tet=skin_tet,
                skin_bary=skin_bary,
                vis_verts=base_vis_verts,
                tri_ids=base_tri_ids,
                color=cfg["color"],
            )
            self.bodies.append(inst)
            vert_offset += n_v

        # Złożenie globalnych tablic ciągłych dla kompilatora Numba
        self.init_pos = np.ascontiguousarray(np.vstack(all_init_pos), dtype=np.float32)
        self.rest_pos = np.ascontiguousarray(np.vstack(all_rest_pos), dtype=np.float32)
        self.pos = self.init_pos.copy()
        self.prev_pos = self.init_pos.copy()
        self.vel = np.zeros_like(self.pos)

        self.body_ids = np.ascontiguousarray(np.concatenate(all_body_ids), dtype=np.int32)
        self.tet_ids = np.ascontiguousarray(np.vstack(all_tet_ids), dtype=np.int32)
        self.edge_ids = np.ascontiguousarray(np.vstack(all_edge_ids), dtype=np.int32)

        self.num_particles = self.pos.shape[0]
        self.num_tets = self.tet_ids.shape[0]

        self.inv_mass = np.zeros(self.num_particles, dtype=np.float32)
        self.radii = np.zeros(self.num_particles, dtype=np.float32)
        self.rest_vol = np.zeros(self.num_tets, dtype=np.float32)
        self.edge_lengths = np.zeros(self.edge_ids.shape[0], dtype=np.float32)

        # Parametry fizyczne XPBD
        self.gravity = np.array([0.0, -10.0, 0.0], dtype=np.float32)
        self.edge_compliance = 0.0
        self.vol_compliance = 0.0
        self.collisions_enabled = True
        self.friction = 0.35
        self.ground_friction = 0.25

        # Interaktywna sztywna kula kolizyjna (włączana klawiszem B)
        self.sphere_active = True
        self.sphere_radius = 0.35
        self.sphere_init_pos = np.array([0.85, 0.35, 0.0], dtype=np.float32)
        self.sphere_pos = self.sphere_init_pos.copy()
        self.sphere_prev_pos = self.sphere_init_pos.copy()

        # Stan chwytaka myszy (Grabber)
        self.grab_id = -1
        self.grab_inv_mass = 0.0

        self._init_physics()

        # Bufory dla Spatial Hasha kolizji (rozmiar komórki = 2 * maksymalny promień cząstki)
        self.hash_spacing = float(2.05 * np.max(self.radii))
        table_size = 4 * self.num_particles + 101
        self.cell_start = np.zeros(table_size + 1, dtype=np.int32)
        self.cell_entries = np.zeros(self.num_particles, dtype=np.int32)

        self.update_visual_meshes()

    def _init_physics(self):
        """Oblicza masy cząstek, spoczynkowe objętości czworościanów oraz promienie kolizyjne."""
        self.inv_mass.fill(0.0)
        vert_vol = np.zeros(self.num_particles, dtype=np.float32)

        p0 = self.pos[self.tet_ids[:, 0]]
        p1 = self.pos[self.tet_ids[:, 1]]
        p2 = self.pos[self.tet_ids[:, 2]]
        p3 = self.pos[self.tet_ids[:, 3]]

        vols = np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0
        self.rest_vol[:] = vols.astype(np.float32)

        for i in range(self.num_tets):
            vol = self.rest_vol[i]
            if vol > 0.0:
                quarter_vol = vol * 0.25
                p_inv_mass = 1.0 / quarter_vol
                for j in range(4):
                    vid = self.tet_ids[i, j]
                    self.inv_mass[vid] += p_inv_mass
                    vert_vol[vid] += quarter_vol

        # Promień kuli kolizyjnej wyprowadzony z objętości przypisanej wierzchołkowi: V = 4/3 * pi * r^3
        raw_radii = np.cbrt((3.0 * np.maximum(vert_vol, 1e-8)) / (4.0 * math.pi)) * 1.15
        self.radii[:] = np.clip(raw_radii, 0.022, 0.052).astype(np.float32)

        e0 = self.pos[self.edge_ids[:, 0]]
        e1 = self.pos[self.edge_ids[:, 1]]
        self.edge_lengths[:] = np.linalg.norm(e0 - e1, axis=1).astype(np.float32)

    def simulate_frame(self, dt: float, num_substeps: int):
        """
        GŁÓWNA PĘTLA SYMULACJI XPBD DLA JEDNEJ KLATKI OBRAZU:
        Dzielimy krok dt (1/60 s) na num_substeps (10) mniejszych podkroków sdt.
        """
        sdt = dt / num_substeps
        for _ in range(num_substeps):
            # KROK 1: Predykcja pozycji z grawitacji + podłoga
            xpbd_step_1_pre_solve(
                self.pos, self.prev_pos, self.vel, self.inv_mass,
                self.radii, self.gravity, sdt, self.ground_friction
            )

            # KROK 2A: Rozwiązywanie więzów krawędzi
            xpbd_step_2a_solve_edges(
                self.pos, self.inv_mass, self.edge_ids,
                self.edge_lengths, self.edge_compliance, sdt
            )

            # KROK 2B: Rozwiązywanie więzów objętości czworościanów
            xpbd_step_2b_solve_volumes(
                self.pos, self.inv_mass, self.tet_ids,
                self.rest_vol, self.VOL_ID_ORDER, self.vol_compliance, sdt
            )

            # KROK 2C: Rozwiązywanie kolizji (między smokami, wewnętrznych oraz z kulą)
            if self.collisions_enabled:
                xpbd_step_2c_solve_collisions(
                    self.pos, self.prev_pos, self.rest_pos, self.inv_mass,
                    self.radii, self.body_ids, self.cell_start, self.cell_entries,
                    self.hash_spacing, self.friction,
                    self.sphere_active, self.sphere_pos, self.sphere_prev_pos, self.sphere_radius
                )

            # KROK 3: Wyznaczenie prędkości z korekt pozycji
            xpbd_step_3_post_solve(self.pos, self.prev_pos, self.vel, self.inv_mass, sdt)

            # Zapamiętanie poprzedniej pozycji kuli dla obliczania tarcia w kolejnym podkroku
            self.sphere_prev_pos[:] = self.sphere_pos

        # Po zakończeniu wszystkich podkroków aktualizujemy gęste siatki wizualne do wyrenderowania
        self.update_visual_meshes()

    def update_visual_meshes(self):
        for b in self.bodies:
            update_vis_mesh_kernel(
                self.pos, b.vert_offset, b.local_tet_ids,
                b.skin_tet, b.skin_bary, b.tri_ids, b.vis_pos, b.vis_normals
            )

    def restart(self):
        self.end_grab(np.zeros(3, dtype=np.float32))
        self.pos[:] = self.init_pos
        self.prev_pos[:] = self.init_pos
        self.vel.fill(0.0)
        self.sphere_pos[:] = self.sphere_init_pos
        self.sphere_prev_pos[:] = self.sphere_init_pos
        self._init_physics()
        self.update_visual_meshes()

    def squash(self):
        """Spłaszcza smoki na wysokości y = 0.5 (rozsunięte w osi X, żeby efekt był czytelny)."""
        self.end_grab(np.zeros(3, dtype=np.float32))
        for b in self.bodies:
            s, e = b.vert_offset, b.vert_offset + b.num_verts
            shift_x = -0.75 if b.body_id == 0 else 0.75
            mean_x = float(np.mean(self.pos[s:e, 0]))
            self.pos[s:e, 0] += (shift_x - mean_x)
            self.pos[s:e, 1] = 0.5
        self.vel.fill(0.0)
        self.update_visual_meshes()

    def start_grab(self, body: SoftBodyInstance, hit_pos: np.ndarray):
        s, e = body.vert_offset, body.vert_offset + body.num_verts
        diff = self.pos[s:e] - hit_pos
        d2 = np.einsum("ij,ij->i", diff, diff)
        self.grab_id = s + int(np.argmin(d2))
        self.grab_inv_mass = float(self.inv_mass[self.grab_id])
        self.inv_mass[self.grab_id] = 0.0
        self.pos[self.grab_id] = hit_pos

    def move_grabbed(self, pos: np.ndarray):
        if self.grab_id >= 0:
            self.pos[self.grab_id] = pos

    def end_grab(self, vel: np.ndarray):
        if self.grab_id >= 0:
            self.inv_mass[self.grab_id] = self.grab_inv_mass
            self.vel[self.grab_id] = vel
            self.grab_id = -1


# ==============================================================================
# CZĘŚĆ 5: KAMERA ORBITALNA, CHWYTAK MYSZY (GRABBER) I RENDERING OPENGL
# ==============================================================================

class OrbitCamera:
    def __init__(self):
        self.target = np.array([0.0, 0.85, 0.0], dtype=np.float32)
        self.distance = 3.8
        self.yaw = 0.25
        self.pitch = 0.22
        self.fov_y = 65.0

    def basis(self):
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        sy, cy = math.sin(self.yaw), math.cos(self.yaw)
        eye = self.target + np.array([self.distance * cp * sy, self.distance * sp, self.distance * cp * cy], dtype=np.float32)
        fwd = self.target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        return eye, fwd, right, up

    def view_matrix(self) -> np.ndarray:
        eye, fwd, right, up = self.basis()
        m = np.eye(4, dtype=np.float32)
        m[0, :3], m[1, :3], m[2, :3] = right, up, -fwd
        m[0, 3], m[1, 3], m[2, 3] = -np.dot(right, eye), -np.dot(up, eye), np.dot(fwd, eye)
        return np.ascontiguousarray(m.T, dtype=np.float32)

    def projection_matrix(self, aspect: float, z_near=0.05, z_far=100.0) -> np.ndarray:
        t = 1.0 / math.tan(math.radians(self.fov_y) * 0.5)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = t / max(aspect, 1e-4)
        m[1, 1] = t
        m[2, 2] = (z_far + z_near) / (z_near - z_far)
        m[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
        m[3, 2] = -1.0
        return np.ascontiguousarray(m.T, dtype=np.float32)

    def ray(self, mx: float, my: float, w: int, h: int):
        eye, fwd, right, up = self.basis()
        nx = (2.0 * mx / max(w, 1)) - 1.0
        ny = 1.0 - (2.0 * my / max(h, 1))
        tan_half = math.tan(math.radians(self.fov_y) * 0.5)
        d = fwd + (nx * (w / max(h, 1)) * tan_half) * right + (ny * tan_half) * up
        d /= np.linalg.norm(d)
        return eye, d.astype(np.float32)


class Grabber:
    """Pozwala łapać myszką zarówno dowolnego smoka, jak i sztywną kulę kolizyjną."""
    def __init__(self):
        self.mode = None  # "soft" lub "sphere"
        self.distance = 0.0
        self.grab_offset = np.zeros(3, dtype=np.float32)
        self.prev_pos = np.zeros(3, dtype=np.float32)
        self.vel = np.zeros(3, dtype=np.float32)
        self.time = 0.0

    def increase_time(self, dt: float):
        self.time += dt

    def start(self, mx, my, w, h, camera: OrbitCamera, world: XPBDWorld) -> bool:
        self.mode = None
        ray_orig, ray_dir = camera.ray(mx, my, w, h)
        min_t = 1e30
        hit_body = None

        for b in world.bodies:
            t = raycast_triangles(ray_orig, ray_dir, b.vis_pos, b.tri_ids)
            if t < min_t:
                min_t = t
                hit_body = b

        t_sphere = 1e30
        if world.sphere_active:
            t_sphere = raycast_sphere(ray_orig, ray_dir, world.sphere_pos, world.sphere_radius)

        if t_sphere < min_t and t_sphere < 1e20:
            self.mode = "sphere"
            self.distance = t_sphere
            hit_pos = ray_orig + t_sphere * ray_dir
            self.grab_offset[:] = world.sphere_pos - hit_pos
            return True

        if hit_body is not None and min_t < 1e20:
            self.mode = "soft"
            self.distance = min_t
            pos = ray_orig + min_t * ray_dir
            world.start_grab(hit_body, pos)
            self.prev_pos[:] = pos
            self.vel.fill(0.0)
            self.time = 0.0
            return True

        return False

    def move(self, mx, my, w, h, camera: OrbitCamera, world: XPBDWorld):
        if self.mode is None:
            return
        ray_orig, ray_dir = camera.ray(mx, my, w, h)
        pos = ray_orig + self.distance * ray_dir

        if self.mode == "sphere":
            new_c = pos + self.grab_offset
            new_c[1] = max(world.sphere_radius, new_c[1])
            world.sphere_pos[:] = new_c
        elif self.mode == "soft":
            if self.time > 0.0:
                self.vel[:] = np.clip((pos - self.prev_pos) / self.time, -18.0, 18.0)
            else:
                self.vel.fill(0.0)
            self.prev_pos[:] = pos
            self.time = 0.0
            world.move_grabbed(pos)

    def end(self, world: XPBDWorld):
        if self.mode == "soft":
            world.end_grab(self.vel)
        self.mode = None


def init_gl():
    glClearColor(0.08, 0.09, 0.11, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)

    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.30, 0.30, 0.32, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.98, 0.96, 0.92, 1.0])
    glLightfv(GL_LIGHT0, GL_SPECULAR, [0.45, 0.45, 0.45, 1.0])

    glLightfv(GL_LIGHT1, GL_AMBIENT, [0.0, 0.0, 0.0, 1.0])
    glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.30, 0.34, 0.42, 1.0])
    glLightfv(GL_LIGHT1, GL_SPECULAR, [0.0, 0.0, 0.0, 1.0])

    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.35, 0.35, 0.35, 1.0])
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, [40.0])


def render_ground():
    glColor3f(0.55, 0.59, 0.62)
    glBegin(GL_TRIANGLES)
    s = 10.0
    glVertex3f(-s, -0.002, -s)
    glVertex3f(-s, -0.002, s)
    glVertex3f(s, -0.002, s)
    glVertex3f(-s, -0.002, -s)
    glVertex3f(s, -0.002, s)
    glVertex3f(s, -0.002, -s)
    glEnd()

    glDisable(GL_LIGHTING)
    glColor3f(0.38, 0.41, 0.44)
    glLineWidth(1.0)
    glBegin(GL_LINES)
    for i in range(-10, 11):
        glVertex3f(float(i), 0.0, -10.0)
        glVertex3f(float(i), 0.0, 10.0)
        glVertex3f(-10.0, 0.0, float(i))
        glVertex3f(10.0, 0.0, float(i))
    glEnd()
    glEnable(GL_LIGHTING)


def render_world(world: XPBDWorld, show_tets: bool):
    glEnableClientState(GL_VERTEX_ARRAY)

    for b in world.bodies:
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        glColor3f(*b.color)

        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, b.vis_pos.ctypes.data_as(ctypes.c_void_p))
        glNormalPointer(GL_FLOAT, 0, b.vis_normals.ctypes.data_as(ctypes.c_void_p))
        glDrawElements(GL_TRIANGLES, b.tri_indices_gl.size, GL_UNSIGNED_INT, b.tri_indices_gl.ctypes.data_as(ctypes.c_void_p))
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisable(GL_POLYGON_OFFSET_FILL)

        if show_tets:
            glDisable(GL_LIGHTING)
            glColor3f(1.0, 1.0, 1.0)
            glLineWidth(1.2)
            glVertexPointer(3, GL_FLOAT, 0, world.pos.ctypes.data_as(ctypes.c_void_p))
            glDrawElements(GL_LINES, b.edge_indices_gl.size, GL_UNSIGNED_INT, b.edge_indices_gl.ctypes.data_as(ctypes.c_void_p))
            glEnable(GL_LIGHTING)

    glDisableClientState(GL_VERTEX_ARRAY)


# ==============================================================================
# CZĘŚĆ 6: PĘTLA GŁÓWNA APLIKACJI I OBSŁUGA ZDARZEŃ
# ==============================================================================

def main():
    tet_mesh, vis_mesh = load_dragon_meshes()
    world = XPBDWorld(tet_mesh, vis_mesh)

    pygame.init()
    width, height = 1280, 820
    pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL | RESIZABLE)
    init_gl()

    vp = glGetIntegerv(GL_VIEWPORT)
    scale_x, scale_y = vp[2] / width, vp[3] / height

    camera = OrbitCamera()
    grabber = Grabber()

    dt = 1.0 / 60.0
    num_substeps = 10
    paused = True
    show_tets = False
    compliance_level = 0

    orbiting = False
    panning = False
    clock = pygame.time.Clock()

    print("Sterowanie symulacją XPBD (wariant z kolizjami na 5 pkt):")
    print("  [SPACE]     : Start / Stop symulacji")
    print("  [LPM obiekt]: Złap i rzuć dowolnego smoka lub czerwoną kulę")
    print("  [LPM tło]   : Obrót kamery | [PPM]: Przesuwanie | [Rolka]: Zoom")
    print("  [C]         : Włącz / Wyłącz obsługę kolizji (dla porównania efektu!)")
    print("  [B]         : Pokaż / Ukryj sztywną kulę kolizyjną")
    print("  [S]         : Squash (zgniecenie smoków)")
    print("  [T]         : Pokaż / Ukryj siatkę czworościanów (tetMesh)")
    print("  [R]         : Restart sceny")
    print("  [UP / DOWN] : Zmiana podatności sprężystej (Compliance 0..10)")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            elif event.type == VIDEORESIZE:
                width, height = max(event.w, 100), max(event.h, 100)
                glViewport(0, 0, int(width * scale_x), int(height * scale_y))

            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_SPACE:
                    paused = not paused
                elif event.key == K_r:
                    world.restart()
                    paused = True
                elif event.key == K_s:
                    world.squash()
                    paused = True
                elif event.key == K_t:
                    show_tets = not show_tets
                elif event.key == K_c:
                    world.collisions_enabled = not world.collisions_enabled
                elif event.key == K_b:
                    world.sphere_active = not world.sphere_active
                elif event.key == K_UP:
                    compliance_level = min(10, compliance_level + 1)
                    world.edge_compliance = compliance_level * 10.0
                elif event.key == K_DOWN:
                    compliance_level = max(0, compliance_level - 1)
                    world.edge_compliance = compliance_level * 10.0

            elif event.type == MOUSEBUTTONDOWN:
                mx, my = event.pos
                if event.button == 1:
                    if grabber.start(mx, my, width, height, camera, world):
                        paused = False
                    else:
                        orbiting = True
                elif event.button == 3:
                    panning = True

            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    grabber.end(world)
                    orbiting = False
                elif event.button == 3:
                    panning = False

            elif event.type == MOUSEMOTION:
                dx, dy = event.rel
                mx, my = event.pos
                if grabber.mode is not None:
                    grabber.move(mx, my, width, height, camera, world)
                elif orbiting:
                    camera.yaw -= dx * 0.007
                    camera.pitch = max(-1.45, min(1.45, camera.pitch + dy * 0.007))
                elif panning:
                    _, _, right, up = camera.basis()
                    s = camera.distance * 0.0015
                    camera.target -= right * (dx * s)
                    camera.target += up * (dy * s)

            elif event.type == MOUSEWHEEL:
                camera.distance = max(0.6, min(20.0, camera.distance * (0.9 ** event.y)))

        if not paused:
            world.simulate_frame(dt, num_substeps)
            grabber.increase_time(dt)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadMatrixf(camera.projection_matrix(width / max(height, 1)))

        glMatrixMode(GL_MODELVIEW)
        glLoadMatrixf(camera.view_matrix())

        glLightfv(GL_LIGHT0, GL_POSITION, [2.5, 4.0, 3.5, 1.0])
        glLightfv(GL_LIGHT1, GL_POSITION, [-2.0, 3.0, -2.0, 0.0])

        render_ground()
        render_world(world, show_tets)

        pygame.display.flip()
        clock.tick(60)

        col_str = "ON" if world.collisions_enabled else "OFF"
        ball_str = "ON" if world.sphere_active else "OFF"
        run_str = "RUN" if not paused else "PAUSED"
        pygame.display.set_caption(
            f"XPBD Soft Body + Collisions [{run_str}] | {clock.get_fps():.0f} FPS | "
            f"Collisions [C]: {col_str} | Ball [B]: {ball_str} | Compliance: {compliance_level}/10 | "
            f"[Space] Start/Stop  [S] Squash  [T] Tets  [R] Reset"
        )

    pygame.quit()


if __name__ == "__main__":
    main()