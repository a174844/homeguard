// 强制卸载所有旧版 Service Worker
self.addEventListener("install", function(e) {
    self.skipWaiting(); // 立即激活新 SW，不等待旧页面关闭
});

self.addEventListener("activate", function(e) {
    e.waitUntil(
        caches.keys().then(function(keys) {
            return Promise.all(keys.map(function(key) {
                return caches.delete(key); // 清空所有缓存
            }));
        }).then(function() {
            return self.clients.claim(); // 立即接管所有页面
        })
    );
});

// 关键：不再缓存任何请求，全部从网络获取
self.addEventListener("fetch", function(e) {
    e.respondWith(fetch(e.request));
});
