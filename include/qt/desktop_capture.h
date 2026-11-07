#pragma once

#include <QObject>
#include <QImage>
#include <QQuickImageProvider>
#include <QTimer>
#include <QVariantList>

class DesktopCapture final : public QObject {
    Q_OBJECT
    Q_PROPERTY(qulonglong frameSequence READ frameSequence NOTIFY frameChanged)
    Q_PROPERTY(int screenWidth READ screenWidth NOTIFY frameChanged)
    Q_PROPERTY(int screenHeight READ screenHeight NOTIFY frameChanged)
    Q_PROPERTY(int cursorX READ cursorX NOTIFY frameChanged)
    Q_PROPERTY(int cursorY READ cursorY NOTIFY frameChanged)
    Q_PROPERTY(bool capturing READ capturing NOTIFY captureStatusChanged)
    Q_PROPERTY(QString captureStatus READ captureStatus NOTIFY captureStatusChanged)
    Q_PROPERTY(QString pointerActionStatus READ pointerActionStatus NOTIFY captureStatusChanged)
    Q_PROPERTY(QVariantList visualGrid READ visualGrid NOTIFY frameChanged)

public:
    explicit DesktopCapture(QObject* parent = nullptr);

    qulonglong frameSequence() const noexcept;
    int screenWidth() const noexcept;
    int screenHeight() const noexcept;
    int cursorX() const noexcept;
    int cursorY() const noexcept;
    bool capturing() const noexcept;
    QString captureStatus() const;
    QString pointerActionStatus() const;
    QImage image() const;
    QVariantList visualGrid() const;

    Q_INVOKABLE void start();
    Q_INVOKABLE void stop();
    Q_INVOKABLE void refreshNow();
    Q_INVOKABLE void movePointerToNormalized(double x, double y);

signals:
    void frameChanged();
    void captureStatusChanged();

private slots:
    void captureFrame();

private:
    QTimer timer_;
    QImage image_;
    QVariantList visual_grid_;
    qulonglong frame_sequence_ = 0;
    int screen_width_ = 0;
    int screen_height_ = 0;
    int cursor_x_ = 0;
    int cursor_y_ = 0;
    bool capturing_ = false;
    QString capture_status_ = QStringLiteral("Capture stopped");
    QString pointer_action_status_ = QStringLiteral("Pointer actuator idle");
};

class DesktopFrameProvider final : public QQuickImageProvider {
public:
    explicit DesktopFrameProvider(const DesktopCapture& capture);

    QImage requestImage(const QString& id, QSize* size,
                        const QSize& requestedSize) override;

private:
    const DesktopCapture& capture_;
};
