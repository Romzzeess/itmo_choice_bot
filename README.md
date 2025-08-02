# itmo\_choice\_bot

Чат-бот в Telegram для абитуриентов — помогает сравнить магистерские программы **AI** и **AI Product** (ИТМО), собрать профиль, спланировать траекторию обучения и получить персональные рекомендации, используя структурированные учебные планы. В качестве «мозга» используется OpenAI API, знание хранится в ArangoDB и Qdrant. Всё развёртывается через Docker Compose.

## Ключевой стек

* Python + `python-telegram-bot`
* OpenAI API (для формирования объяснений / классификации / уточнений)
* ArangoDB (хранение программ, курсов, профилей, сессий)
* Qdrant (векторное хранилище для семантического поиска/сопоставления)
* Docker Compose — всё в контейнерах
* Рекомендуемое железо: **NVIDIA GeForce RTX 3060** (см. раздел ниже)

## Требования

1. **Аппаратное**

   * Минимум **NVIDIA GeForce RTX 3060** или лучше, если планируются локальные модели.
   * Установленные драйверы NVIDIA и **NVIDIA Container Toolkit** (для доступа контейнеров к GPU).

2. **Софт**

   * Docker (>=20.10)
   * Docker Compose (v2+)
   * Telegram-бот токен
   * OpenAI API ключ (Если нет видеокарты)

## Быстрый старт

### 1. Клонируй репозиторий

```bash
git clone https://github.com/Romzzeess/itmo_choice_bot.git
cd itmo_choice_bot
```

### 2. Создай `.env` файл (по образцу)

Скопируй шаблон и заполните:

```bash
cp .env.example .env
```

Пример содержания `.env` (вставь реальные значения):

```
TELEGRAM_TOKEN=123456:ABCDEF...
OPENAI_API_KEY=sk-...
ARANGODB_URL=http://arangodb:8529
ARANGODB_ROOT_PASSWORD=yourpassword
QDRANT_URL=http://qdrant:6333
```

### 3. Убедись, что доступен NVIDIA GPU (если нужен)

Если ты используешь GPU (рекомендуется для будущих расширений), установи:

* NVIDIA драйверы
* NVIDIA Container Toolkit: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Проверь доступность в Docker:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
```

### 4. Запусти стек через Docker Compose

```bash
docker compose up --build
```

Это поднимет:

* Triton (Triton client с моделью эмбеддингов)
* SGLang (OpenAI клиент с моделью Qwen3-4B)
* ArangoDB (сохранение знаний / профилей)
* Qdrant (векторное хранилище)
