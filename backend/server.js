const express = require('express');
const multer = require('multer');
const cors = require('cors');
const bodyParser = require('body-parser');
const { exec } = require('child_process');
const path = require('path');

const app = express();
const PORT = 5001;

// 配置 CORS 和解析 JSON
app.use(cors());
app.use(bodyParser.json());

// 配置静态文件服务
app.use('/styles', express.static(path.join(__dirname, 'styles')));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

const fs = require("fs");

// 新增 API：获取样式列表
app.get("/styles/list", (req, res) => {
    const stylesDir = path.join(__dirname, "styles");
    fs.readdir(stylesDir, (err, files) => {
        if (err) {
            console.error("Error reading styles directory:", err);
            return res.status(500).json({ error: "Failed to load styles" });
        }

        // 过滤出 .jpg 文件并移除扩展名
        const styles = files
            .filter((file) => file.endsWith(".jpg"))
            .map((file) => path.basename(file, ".jpg")); // 移除扩展名

        res.json(styles); // 返回样式名称列表
    });
});

// 配置文件上传
const storage = multer.diskStorage({
    destination: './uploads/',
    filename: (req, file, cb) => {
        const ext = path.extname(file.originalname).toLowerCase();
        cb(null, Date.now() + ext);
    }
});
const upload = multer({ storage });

// 上传图片接口
app.post('/upload', upload.single('image'), (req, res) => {
    const imagePath = req.file.path;
    const style = req.body.style; // 用户选择的风格

    console.log(`Image uploaded: ${imagePath}`);
    console.log(`Selected style: ${style}`);

    // 调用 Python 脚本进行风格迁移
    const outputImagePath = `uploads/output_${Date.now()}.jpg`; // 确保相对路径
    const command = `/Users/dwx/Library/Caches/pypoetry/virtualenvs/7370env-6DOi-_x6-py3.12/bin/python style_transfer.py --content ${imagePath} --style ./styles/${style}.jpg --output ${outputImagePath}`;

    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error("Error executing style transfer script:", stderr);
            return res.status(500).json({
                error: 'Style transfer failed',
                details: stderr || error.message
            });
        }

        console.log("Style transfer output:", stdout);
        res.json({ imagePath: `/${outputImagePath}` }); // 前端需要加 `/`
    });
});

// 启动服务器
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
