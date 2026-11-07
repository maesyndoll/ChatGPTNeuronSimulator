#include "qt/desktop_capture.h"

#include <QCursor>
#include <QGuiApplication>
#include <QPixmap>
#include <QScreen>
#include <QColor>

#include <algorithm>

DesktopCapture::DesktopCapture(QObject* parent)
    : QObject(parent) {
    timer_.setInterval(500); // 2 Hz input sampling; the dashboard is a neural monitor, not a video player.
    connect(&timer_, &QTimer::timeout,
            this, &DesktopCapture::captureFrame);
}

qulonglong DesktopCapture::frameSequence() const noexcept {
    return frame_sequence_;
}

int DesktopCapture::screenWidth() const noexcept {
    return screen_width_;
}

int DesktopCapture::screenHeight() const noexcept {
    return screen_height_;
}

int DesktopCapture::cursorX() const noexcept {
    return cursor_x_;
}

int DesktopCapture::cursorY() const noexcept {
    return cursor_y_;
}

bool DesktopCapture::capturing() const noexcept {
    return capturing_;
}

QString DesktopCapture::captureStatus() const {
    return capture_status_;
}

QImage DesktopCapture::image() const {
    return image_;
}

QVariantList DesktopCapture::visualGrid() const {
    return visual_grid_;
}

void DesktopCapture::start() {
    if (capturing_)
        return;

    capturing_ = true;
    capture_status_ = QStringLiteral("Capturing primary display");
    timer_.start();
    emit captureStatusChanged();
    captureFrame();
}

void DesktopCapture::stop() {
    if (!capturing_)
        return;

    timer_.stop();
    capturing_ = false;
    capture_status_ = QStringLiteral("Capture stopped");
    emit captureStatusChanged();
}

void DesktopCapture::refreshNow() {
    if (capturing_)
        captureFrame();
}

void DesktopCapture::captureFrame() {
    QScreen* screen = QGuiApplication::primaryScreen();
    if (screen == nullptr) {
        capture_status_ = QStringLiteral("No primary screen is available");
        emit captureStatusChanged();
        return;
    }

    const QRect geometry = screen->geometry();
    const QPixmap pixmap = screen->grabWindow(0);
    if (pixmap.isNull()) {
        capture_status_ = QStringLiteral(
            "Screen capture returned an empty frame; compositor permissions may be required");
        emit captureStatusChanged();
        return;
    }

    image_ = pixmap.toImage().convertToFormat(QImage::Format_Grayscale8);
    const QImage reduced = image_.scaled(8, 8, Qt::IgnoreAspectRatio,
                                         Qt::SmoothTransformation);
    visual_grid_.clear();
    visual_grid_.reserve(64);
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            const int luminance = qGray(reduced.pixel(x, y));
            visual_grid_.append(static_cast<double>(luminance) / 255.0);
        }
    }
    screen_width_ = image_.width();
    screen_height_ = image_.height();

    const QPoint cursor = QCursor::pos() - geometry.topLeft();
    cursor_x_ = cursor.x();
    cursor_y_ = cursor.y();

    ++frame_sequence_;
    capture_status_ = QStringLiteral("Live · %1 × %2")
        .arg(screen_width_).arg(screen_height_);
    emit frameChanged();
    emit captureStatusChanged();
}

DesktopFrameProvider::DesktopFrameProvider(const DesktopCapture& capture)
    : QQuickImageProvider(QQuickImageProvider::Image),
      capture_(capture) {
}

QImage DesktopFrameProvider::requestImage(
    const QString&,
    QSize* size,
    const QSize&) {
    QImage result = capture_.image();
    if (result.isNull()) {
        result = QImage(960, 540, QImage::Format_RGB32);
        result.fill(qRgb(10, 17, 27));
    }
    if (size != nullptr)
        *size = result.size();
    return result;
}
