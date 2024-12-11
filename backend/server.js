const express = require('express');
const multer = require('multer');
const cors = require('cors');
const bodyParser = require('body-parser');
const { exec } = require('child_process');
const path = require('path');
const sharp = require('sharp');
const fs = require('fs');

const app = express();
const PORT = 5001;

// 配置 CORS 和解析 JSON
app.use(cors());
app.use(bodyParser.json());

// 配置静态文件服务
app.use(
    '/styles',
    express.static(path.join(__dirname, 'styles'), {
        setHeaders: (res, path) => {
            if (path.endsWith('.jpg')) {
                res.setHeader('Content-Type', 'image/jpeg');
            }
        },
    })
);
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));
app.use('/output', express.static(path.join(__dirname, 'output'))); // 新增输出文件夹静态服务

// 确保 output 文件夹存在
const outputDir = path.join(__dirname, 'output');
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir);
}


// 生成缩略图函数
async function generateThumbnail(inputPath, outputPath) {
    try {
        await sharp(inputPath)
            .resize(200, 200, { // 提高分辨率
                fit: 'cover',
            })
            .toFile(outputPath);
        console.log(`Thumbnail generated: ${outputPath}`);
    } catch (error) {
        console.error('Error generating thumbnail:', error);
    }
}

// 返回样式列表
app.get('/styles/list', (req, res) => {
    const stylesDir = path.join(__dirname, 'styles');
    const modelsDir = path.join(__dirname, 'models');

    // 获取所有模型文件名（去除扩展名）
    const modelFiles = fs
        .readdirSync(modelsDir)
        .filter((file) => file.endsWith('.model'))
        .map((file) => path.basename(file, '.model'));

    fs.readdir(stylesDir, async (err, files) => {
        if (err) {
            console.error('Error reading styles directory:', err);
            return res.status(500).json({ error: 'Failed to load styles' });
        }

        const styles = [];
        for (const model of modelFiles) {
            const imageFile = files.find(
                (file) =>
                    path.basename(file, path.extname(file)).toLowerCase() ===
                    model.toLowerCase()
            );
            if (imageFile) {
                const styleName = model;
                const thumbnailPath = path.join(stylesDir, `${styleName}_thumbnail.jpg`);

                // 检查是否已有缩略图，没有则生成
                if (!fs.existsSync(thumbnailPath)) {
                    await generateThumbnail(
                        path.join(stylesDir, imageFile),
                        thumbnailPath
                    );
                }

                styles.push({
                    styleName,
                    imagePath: `/styles/${imageFile}`,
                    thumbnailPath: `/styles/${styleName}_thumbnail.jpg`,
                });
            }
        }
        console.log('Styles to send:', styles); // 调试输出
        res.json(styles);
    });
});

// 配置文件上传
const storage = multer.diskStorage({
    destination: './uploads/',
    filename: (req, file, cb) => {
        const ext = path.extname(file.originalname).toLowerCase();
        cb(null, Date.now() + ext);
    },
});
const upload = multer({ storage });

// 上传图片接口
app.post('/upload', upload.single('image'), (req, res) => {
    const imagePath = req.file.path;
    const style = req.body.style; // 用户选择的风格

    console.log(`Image uploaded: ${imagePath}`);
    console.log(`Selected style: ${style}`);

    // 确保输出目录存在
    const outputDir = path.join(__dirname, 'output');
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir);
    }

    const outputImagePath = `output/output_${Date.now()}.jpg`;
    const command = `/Users/dwx/Library/Caches/pypoetry/virtualenvs/7370env-6DOi-_x6-py3.12/bin/python style.py transfer --model-path ./models/${style}.model --source ${imagePath} --output ${outputImagePath}`;

    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error('Error executing style transfer script:', stderr);
            return res.status(500).json({
                error: 'Style transfer failed',
                details: stderr || error.message,
            });
        }

        console.log('Style transfer output:', stdout);
        res.json({ imagePath: `/${outputImagePath}` }); // 更新路径
    });
});


// Blend functionality
app.post('/blend', upload.single('image'), (req, res) => {
    const { style1, style2, weight } = req.body;

    if (!style1 || !style2 || !weight) {
        return res.status(400).json({ error: "Missing required parameters: style1, style2, weight" });
    }

    const blendWeight = parseFloat(weight);
    if (isNaN(blendWeight) || blendWeight < 0 || blendWeight > 1) {
        return res.status(400).json({ error: "Invalid blend weight. Must be a float between 0 and 1." });
    }

    const imagePath = req.file.path;
    const outputImagePath = `output/blended_output_${Date.now()}.jpg`;
    const command = `python style.py blend --model-path1 ./models/${style1}.model --model-path2 ./models/${style2}.model --weight ${blendWeight} --source ${imagePath} --output ${outputImagePath}`;

    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error("Error executing blend script:", stderr);
            return res.status(500).json({
                error: "Blending styles failed",
                details: stderr || error.message,
            });
        }

        console.log("Blend styles output:", stdout);
        res.json({ imagePath: `/${outputImagePath}` });
    });
});



// 启动服务器
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});


