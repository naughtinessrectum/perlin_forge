import io
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import cpu_count
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# карта градиента: список (позиция 0‒1, (R, G, B) 0‒255)
GradientMap = list[tuple[float, tuple[int, int, int]]]


# шум Перлина

def _fade(t: np.ndarray) -> np.ndarray:
    """
    квинтичная кривая сглаживания по формуле 6t⁵ − 15t⁴ + 10t³.

    Args:
        t (np.ndarray): входной массив значений, обычно в диапазоне [0, 1].

    Note:
        функция используется для создания плавных переходов, например, в алгоритме шума Перлина.

    Returns:
        np.ndarray: массив со значениями после применения функции сглаживания.
    """
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _perlin_2d(
        shape: tuple[int, int],
        res: tuple[int, int],
        rng: np.random.Generator,
) -> np.ndarray:
    """
    генерирует одну октаву двумерного шума перлина.

    Args:
        shape (tuple[int, int]): кортеж `(height, width)`, определяющий форму выходного массива.
        res (tuple[int, int]): кортеж `(ry, rx)`, задающий разрешение сетки градиентов.
            определяет 'масштаб' или 'частоту' генерируемого шума.
        rng (np.random.Generator): экземпляр генератора случайных чисел numpy для
            обеспечения воспроизводимости.

    Note:
        функция является низкоуровневой и генерирует только один 'слой' шума.
        для получения классического фрактального шума перлина требуется
        суммирование нескольких октав с разным разрешением и амплитудой.

    Returns:
        np.ndarray: двумерный массив numpy сгенерированного шума. значения
            находятся в диапазоне приблизительно [-sqrt(2)/2, sqrt(2)/2].
    """
    h, w = shape
    ry, rx = res

    # случайные единичные градиенты в узлах сетки
    angles = rng.uniform(0, 2 * np.pi, (ry + 1, rx + 1))
    gy = np.sin(angles)
    gx = np.cos(angles)

    # координаты каждого пикселя в пространстве сетки
    yc = np.linspace(0, ry, h, endpoint=False)
    xc = np.linspace(0, rx, w, endpoint=False)

    y0 = np.floor(yc).astype(int)
    x0 = np.floor(xc).astype(int)
    dy = yc - y0  # дробная часть [0, 1)
    dx = xc - x0

    fy = _fade(dy)
    fx = _fade(dx)

    # 2-D broadcasting: строки ↓  столбцы →
    Y0 = y0[:, None]
    X0 = x0[None, :]
    DY = dy[:, None]
    DX = dx[None, :]
    FY = fy[:, None]
    FX = fx[None, :]

    # dot(gradient, offset) в четырёх углах ячейки
    d00 = gy[Y0, X0] * DY + gx[Y0, X0] * DX
    d10 = gy[Y0 + 1, X0] * (DY - 1) + gx[Y0 + 1, X0] * DX
    d01 = gy[Y0, X0 + 1] * DY + gx[Y0, X0 + 1] * (DX - 1)
    d11 = gy[Y0 + 1, X0 + 1] * (DY - 1) + gx[Y0 + 1, X0 + 1] * (DX - 1)

    # билинейная интерполяция с fade
    top = d00 + FY * (d10 - d00)
    bot = d01 + FY * (d11 - d01)
    return top + FX * (bot - top)


def _fbm(
        h: int, w: int,
        rng: np.random.Generator,
        *,
        octaves: int = 6,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
        base_res: int = 4,
) -> np.ndarray:
    """
    генерирует двумерный фрактальный броуновский шум (fbm).

    алгоритм создает шум путем суммирования нескольких слоев (октав) шума перлина
    с возрастающей частотой и убывающей амплитудой. результат представляет собой
    текстуру, часто используемую для процедурной генерации. выходной массив
    нормализуется в диапазон [0, 1].

    Args:
        h (int): высота генерируемого массива шума.
        w (int): ширина генерируемого массива шума.
        rng (np.random.Generator): экземпляр генератора случайных чисел numpy.
        octaves (int, optional): количество октав шума. большее значение
            добавляет больше деталей. по умолчанию 6.
        persistence (float, optional): множитель, контролирующий уменьшение
            амплитуды для каждой следующей октавы. по умолчанию 0.5.
        lacunarity (float, optional): множитель, контролирующий увеличение
            частоты для каждой следующей октавы. по умолчанию 2.0.
        base_res (int, optional): базовое разрешение для первой октавы шума.
            по умолчанию 4.
    """
    out = np.zeros((h, w))
    amp, total = 1.0, 0.0

    for i in range(octaves):
        r = max(1, min(int(round(base_res * lacunarity ** i)), max(h, w)))
        out += _perlin_2d((h, w), (r, r), rng) * amp
        total += amp
        amp *= persistence

    out /= total
    lo, hi = out.min(), out.max()
    return (out - lo) / (hi - lo + 1e-12)


# карта градиента

def _apply_gradient(gray: np.ndarray, stops: GradientMap) -> np.ndarray:
    """
    колоризует монохромное изображение путем линейной интерполяции по градиентной карте.

    Args:
        gray (np.ndarray): входное монохромное изображение в виде numpy массива
            со значениями в диапазоне [0, 1].
        stops (GradientMap): градиентная карта в виде списка контрольных точек.
            каждая точка — это кортеж `(pos, (r, g, b))`, где `pos` — позиция
            на градиенте (float от 0 до 1), а `(r, g, b)` — цвет в формате
            RGB (int от 0 до 255).

    Returns:
        np.ndarray: цветное изображение в формате RGB в виде numpy массива
            со значениями в диапазоне [0, 1].
    """
    stops = sorted(stops, key=lambda s: s[0])
    positions = np.array([s[0] for s in stops])
    colours = np.array([s[1] for s in stops], dtype=np.float64) / 255.0

    rgb = np.empty((*gray.shape, 3))
    for c in range(3):
        rgb[..., c] = np.interp(gray, positions, colours[:, c])
    return rgb


# генератор

def generate_random_image(
        width: int,
        height: int,
        *,
        # переключатели слоёв
        enable_clouds: bool = True,
        enable_rgb_noise: bool = True,
        enable_grain: bool = True,
        # карты градиента (None → монохром)
        cloud_gradient: GradientMap | None = None,
        grain_gradient: GradientMap | None = None,
        # параметры облаков
        cloud_base_res: int = 4,
        cloud_octaves: int = 6,
        cloud_persistence: float = 0.5,
        cloud_lacunarity: float = 2.0,
        # непрозрачность наложения
        rgb_noise_opacity: float = 0.30,
        grain_opacity: float = 0.20,
        # воспроизводимость
        seed: int | None = None,
) -> Image.Image:
    """
    генерирует случайное процедурное текстурное изображение.

    изображение создается путем последовательного наложения нескольких слоев:
    облака (на основе шума перлина), цветной rgb-шум и монохромное зерно.
    каждый слой можно настроить или отключить.

    Args:
        width (int): размер изображения по ширине.
        height (int): размер изображения по высоте.
        enable_clouds (bool, optional): флаг включения слоя облаков.
            по умолчанию True.
        enable_rgb_noise (bool, optional): флаг включения слоя цветного rgb-шума.
            по умолчанию True.
        enable_grain (bool, optional): флаг включения слоя монохромного зерна.
            по умолчанию True.
        cloud_gradient (GradientMap | None, optional): карта градиента для слоя
            облаков. если None, слой будет монохромным. по умолчанию None.
        grain_gradient (GradientMap | None, optional): карта градиента для слоя
            зерна. если None, слой будет монохромным. по умолчанию None.
        cloud_base_res (int, optional): базовое разрешение сетки шума перлина.
            большее значение создает более мелкий узор. по умолчанию 4.
        cloud_octaves (int, optional): количество октав для генерации
            фрактального шума (fBm). по умолчанию 6.
        cloud_persistence (float, optional): коэффициент затухания амплитуды
            для каждой следующей октавы (0-1). по умолчанию 0.5.
        cloud_lacunarity (float, optional): множитель частоты для каждой
            следующей октавы. по умолчанию 2.0.
        rgb_noise_opacity (float, optional): непрозрачность слоя цветного шума
            (0-1). по умолчанию 0.30.
        grain_opacity (float, optional): непрозрачность слоя монохромного зерна
            (0-1). по умолчанию 0.20.
        seed (int | None, optional): начальное значение для генератора случайных
            чисел для обеспечения воспроизводимости результата. по умолчанию None.

    Note:
        первый включенный слой становится базовым, последующие накладываются
        поверх него с указанной непрозрачностью.

    Returns:
        Image.Image: сгенерированное изображение в формате PIL (режим RGB,
            8-бит на канал).
    """
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2 ** 31))
    rng = np.random.default_rng(seed)

    canvas: np.ndarray | None = None  # float64 [0, 1], (H, W, 3)

    # облака (Perlin fBm)
    if enable_clouds:
        gray = _fbm(
            height, width, rng,
            octaves=cloud_octaves,
            persistence=cloud_persistence,
            lacunarity=cloud_lacunarity,
            base_res=cloud_base_res,
        )
        layer = _apply_gradient(gray, cloud_gradient) if cloud_gradient else \
            np.stack([gray] * 3, axis=-1)
        canvas = layer

    # цветной RGB-шум
    if enable_rgb_noise:
        noise = rng.random((height, width, 3))

        # σ пропорционально диагонали: ≈ 1 px при стороне 1500 px,
        # ± 50 % случайного разброса
        diag = np.hypot(width, height)
        sigma = (diag / 1500.0) * rng.uniform(0.5, 1.5)
        for c in range(3):
            noise[:, :, c] = gaussian_filter(noise[:, :, c], sigma=sigma)

        if canvas is None:
            canvas = noise
        else:
            canvas = canvas * (1 - rgb_noise_opacity) + noise * rgb_noise_opacity

    # монохромное зерно
    if enable_grain:
        gray = rng.random((height, width))
        layer = _apply_gradient(gray, grain_gradient) if grain_gradient else \
            np.stack([gray] * 3, axis=-1)

        if canvas is None:
            canvas = layer
        else:
            canvas = canvas * (1 - grain_opacity) + layer * grain_opacity

    # пустой холст, если все слои выключены
    if canvas is None:
        canvas = np.zeros((height, width, 3))

    return Image.fromarray(
        np.clip(canvas * 255, 0, 255).astype(np.uint8), 'RGB'
    )


def get_random_size() -> tuple[int, int, str, str]:
    """
    генерирует случайные размеры изображения на основе стандартных разрешений и соотношений сторон.

    Note:
        длинная сторона изображения соответствует базовому размеру выбранного уровня разрешения.

    Returns:
        кортеж из четырех элементов: ширина (int), высота (int),
        название соотношения сторон (str) и название уровня разрешения (str).
    """
    res_tiers = {
        '0.5K': random.choice((682, 768, 832, 896, 960)),
        '1K': random.choice((1024, 1116, 1152, 1536, 1696, 1792)),
        '2K': random.choice((2000, 2048, 2400, 2688)),
        '3K': random.choice((3072, 3584, 3692)),
        '4K': random.choice((4096, 4096, 4800, 5504))
    }
    ratios = {
        '3:4': (3, 4), '4:3': (4, 3),
        '2:3': (2, 3), '3:2': (3, 2),
        '9:16': (9, 16), '16:9': (16, 9),
        '4:5': (4, 5), '5:4': (5, 4),
        '1:1': (1, 1),
    }

    tier_name, base_dim = random.choice(list(res_tiers.items()))
    ratio_name, (rw, rh) = random.choice(list(ratios.items()))

    # base_dim становится максимальной (длинной) стороной изображения
    if rw > rh:
        width = base_dim
        height = int(base_dim * (rh / rw))
    else:
        height = base_dim
        width = int(base_dim * (rw / rh))

    return width, height, ratio_name, tier_name


def apply_jpeg_degradation(img: Image.Image) -> tuple[Image.Image, int, int]:
    """
    применяет к изображению от 1 до 4 последовательных циклов jpeg-сжатия.

    функция имитирует артефакты, возникающие при многократном пересохранении
    изображения в формате jpeg с низким качеством. количество проходов
    (от 1 до 4) и уровень качества (от 10 до 40) выбираются случайно.

    Args:
        img (Image.Image): исходное изображение для деградации.

    Returns:
        tuple[Image.Image, int, int]: кортеж, содержащий:
            - деградированное изображение (Image.Image).
            - количество выполненных проходов сжатия (int).
            - использованный уровень качества jpeg (int).
    """
    passes = random.randint(1, 4)
    quality = random.randint(10, 40)

    current_img = img
    # сжимаем и распаковываем картинку в оперативной памяти (без сохранения на диск)
    for _ in range(passes):
        buffer = io.BytesIO()
        current_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        current_img = Image.open(buffer).convert('RGB')

    return current_img, passes, quality


def generate_random_gradient(sort_by_luminance: bool = True) -> list[tuple[float, tuple[int, int, int]]]:
    """
    генерация случайной карты градиента с количеством точек от 2 до 5.

    автоматически устанавливает крайние точки на позициях 0.0 и 1.0. при включении
    параметра сортировки цвета упорядочиваются по воспринимаемой яркости (luma),
    что минимизирует резкие визуальные скачки в переходах.

    Args:
        sort_by_luminance (bool, optional): определяет необходимость сортировки цветов
            по яркости. по умолчанию True.

    Returns:
        list[tuple[float, tuple[int, int, int]]]: список контрольных точек, где каждая
        точка — это кортеж из позиции (0.0-1.0) и rgb-цвета.
    """
    num_stops = random.randint(2, 5)

    # гарантируем наличие крайних точек
    positions = [0.0, 1.0]
    for _ in range(num_stops - 2):
        positions.append(random.uniform(0.1, 0.9))
    positions.sort()

    # генерируем случайные цвета
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
              for _ in range(num_stops)]

    if sort_by_luminance:
        # сортировка по воспринимаемой яркости (Luma) для плавных переходов
        colors.sort(key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])

    return [(float(pos), (int(r), int(g), int(b))) for pos, (r, g, b) in zip(positions, colors)]


def _worker_generate(task: dict) -> tuple[str, dict] | None:
    """
    генерирует одно изображение, сохраняет на диск и возвращает
    (filename, metadata).  запускается в дочернем процессе.

    вся случайность уже зафиксирована в `task` — воркер детерминистичен.
    """
    try:
        gen_params = task['gen_params']
        img = generate_random_image(**gen_params)

        # jpeg-деградация (параметры предопределены)
        jpeg_cfg = task.get('jpeg_config')
        if jpeg_cfg:
            current = img
            for _ in range(jpeg_cfg['passes']):
                buf = io.BytesIO()
                current.save(buf, format='JPEG', quality=jpeg_cfg['quality'])
                buf.seek(0)
                current = Image.open(buf).convert('RGB')
            img = current

        # сохранение
        out_path = Path(task['output_dir']) / task['filename']
        img.save(out_path, **task['save_kwargs'])

        return task['filename'], task['metadata']

    except Exception as e:
        print(f"\nошибка при генерации {task.get('filename', '?')}: {e}")
        return None


def create_ml_dataset(
        num_images: int = 1000,
        output_dir: Path | str = 'random_noisy_textures',
        max_workers: int | None = None,
):
    """
    параллельная генерация синтетического датасета изображений и сопутствующих метаданных в формате JSON.

    функция создает набор изображений со случайными визуальными параметрами (разрешение, пропорции,
    шумы, градиенты) и сохраняет подробные характеристики каждого экземпляра в файл метаданных.

    Args:
        num_images (int): количество генерируемых изображений. по умолчанию 1000. (optional)
        output_dir (Path | str): путь к директории для сохранения результатов.
            по умолчанию 'random_noisy_textures'. (optional)
        max_workers (int| None):  количество процессов (optional)

    Note:
        - для каждого изображения случайно выбирается комбинация слоев: облака, RGB-шум и зернистость.
        - в 30% случаев к изображениям применяется симуляция сильного JPEG-сжатия.
        - метаданные сохраняются в файл `_metadata.json` в корне указанной директории.
        - процесс включает автоматическое создание родительских директорий, если они отсутствуют.

    Returns:
        None.

    Raises:
        Exception: логирует ошибки генерации конкретных изображений в консоль, не прерывая общий цикл.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if max_workers is None:
        logical_cpus = cpu_count()
        max_workers = max(1, logical_cpus // 2) if logical_cpus else 4

    # формирование всех задач
    tasks: list[dict] = []

    for i in range(num_images):
        # случайное разрешение и пропорции
        width, height, aspect_ratio, res_tier = get_random_size()

        # рандомизация слоев (гарантия, что хотя бы один включен)
        layers = (False, False, False)
        while not any(layers):
            layers = tuple(random.choice([True, False]) for _ in range(3))
        enable_clouds, enable_rgb_noise, enable_grain = layers
        # рандомизация градиентов
        cloud_grad = None
        if enable_clouds and random.random() > 0.3:
            cloud_grad = generate_random_gradient(
                sort_by_luminance=random.choice([True, False])
            )
        grain_grad = None
        if enable_grain and random.random() > 0.3:
            grain_grad = generate_random_gradient(
                sort_by_luminance=random.choice([True, False])
            )
        # сбор всех параметров
        gen_params = dict(
            width=width,
            height=height,
            enable_clouds=enable_clouds,
            enable_rgb_noise=enable_rgb_noise,
            enable_grain=enable_grain,
            cloud_gradient=cloud_grad,
            grain_gradient=grain_grad,
            cloud_base_res=random.randint(2, 12),
            cloud_octaves=random.randint(2, 8),
            cloud_persistence=round(random.uniform(0.2, 0.8), 3),
            cloud_lacunarity=round(random.uniform(1.5, 3.0), 3),
            rgb_noise_opacity=round(random.uniform(0.05, 0.9), 3),
            grain_opacity=round(random.uniform(0.05, 0.7), 3),
            seed=random.randint(0, 2 ** 31 - 1),
        )

        # jpeg-деградация (шанс 30%)
        do_jpeg = random.random() < 0.30
        jpeg_config = None

        if do_jpeg:
            passes = random.randint(1, 4)
            quality = random.randint(10, 40)
            jpeg_config = {'passes': passes, 'quality': quality}
            filename = f'tex_{i:04d}.jpg'
            save_kwargs = {'format': 'JPEG', 'quality': quality}
        else:
            filename = f'tex_{i:04d}.png'
            save_kwargs = {'format': 'PNG'}

        metadata = {
            **gen_params,
            'aspect_ratio': aspect_ratio,
            'resolution_tier': res_tier,
            'jpeg_artifacts': jpeg_config,
        }

        tasks.append({
            'gen_params': gen_params,
            'jpeg_config': jpeg_config,
            'filename': filename,
            'save_kwargs': save_kwargs,
            'output_dir': str(output_dir),
            'metadata': metadata,
        })

    # параллельная генерация
    all_metadata: dict[str, dict] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker_generate, t): t for t in tasks}

        for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f'генерация (×{max_workers} процессов)',
        ):
            result = future.result()
            if result:
                fname, meta = result
                all_metadata[fname] = meta

    # сохранение метаданных
    meta_path = output_dir / '_metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    print(f'готово: {len(all_metadata)} изображений → "{output_dir}/"')


if __name__ == '__main__':
    try:
        create_ml_dataset(
            num_images=1000,
            output_dir='random_noisy_textures'
        )
    except KeyboardInterrupt:
        print('генерация прервана')
