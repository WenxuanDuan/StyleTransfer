import React, { useState, useEffect } from "react";
import { LazyLoadImage } from "react-lazy-load-image-component";
import "./App.css";

const App = () => {
    const [styles, setStyles] = useState([]);
    const [selectedStyle, setSelectedStyle] = useState("");
    const [selectedFile, setSelectedFile] = useState(null);
    const [uploadedImage, setUploadedImage] = useState(null);
    const [outputImage, setOutputImage] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isHovered, setIsHovered] = useState(false); // 控制蒙版显示状态

    useEffect(() => {
        const fetchStyles = async () => {
            try {
                const response = await fetch("http://localhost:5001/styles/list");
                const data = await response.json();
                setStyles(data);
                if (data.length > 0) setSelectedStyle(data[0].styleName);
            } catch (error) {
                console.error("Error fetching styles:", error);
            }
        };
        fetchStyles();
    }, []);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
        setUploadedImage(URL.createObjectURL(file)); // 生成文件的临时 URL
        setOutputImage(null); // 重置输出图片
    };

    const handleStyleSelect = (styleName) => {
        setSelectedStyle(styleName);
    };

    const handleApplyStyle = async () => {
        if (!selectedFile) {
            alert("Please upload an image!");
            return;
        }
        if (!selectedStyle) {
            alert("Please select a style!");
            return;
        }

        const formData = new FormData();
        formData.append("image", selectedFile);
        formData.append("style", selectedStyle);

        try {
            setIsLoading(true); // 开始加载
            const response = await fetch("http://localhost:5001/upload", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Failed to apply style: ${response.statusText}`);
            }

            const data = await response.json();
            console.log("Response data:", data);

            if (data.imagePath) {
                setOutputImage(`http://localhost:5001${data.imagePath}`); // 设置输出图片路径
            } else {
                throw new Error("Invalid response from server.");
            }
        } catch (error) {
            console.error("Error applying style:", error);
            alert("Failed to apply style. Please try again.");
        } finally {
            setIsLoading(false); // 停止加载
        }
    };

    const handleDownload = async () => {
        if (!outputImage) {
            alert("No styled image available to download.");
            return;
        }

        try {
            const response = await fetch(outputImage);
            if (!response.ok) {
                throw new Error(`Failed to fetch image: ${response.statusText}`);
            }
            const blob = await response.blob(); // 将响应转为 Blob 对象
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob); // 创建临时 URL
            link.download = "styled_image.jpg"; // 设置下载文件名
            document.body.appendChild(link);
            link.click(); // 触发下载
            document.body.removeChild(link); // 移除临时链接
        } catch (error) {
            console.error("Error downloading image:", error);
            alert("Failed to download the styled image.");
        }
    };

    return (
        <div className="App">
            <div className="app-container">
                <h1 className="app-title">
                    🎨 <span>Style Transfer Application</span>
                </h1>

                <div className="upload-section">
                    <h5>Upload Your Image</h5>
                    <input type="file" onChange={handleFileChange} />
                </div>

                <div className="image-display">
                    {/* 上传图片 */}
                    {uploadedImage && (
                        <div className="image-container">
                            <h6>Uploaded Image:</h6>
                            <img src={uploadedImage} alt="Uploaded Preview" />
                        </div>
                    )}

                    {/* 输出图片 */}
                    {outputImage && (
                        <div
                            className="image-container"
                            onMouseEnter={() => setIsHovered(true)} // 显示蒙版
                            onMouseLeave={() => setIsHovered(false)} // 隐藏蒙版
                        >
                            <h6>Styled Image:</h6>
                            <div className="output-image-wrapper">
                                <img src={outputImage} alt="Styled Output" />
                                {isHovered && (
                                    <div
                                        className="download-overlay"
                                        onClick={handleDownload}
                                    >
                                        <i className="fas fa-download"></i> Download
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                <div className="style-section">
                    <h6>Select a Style</h6>
                    <div className="style-grid">
                        {styles.map(({ styleName, imagePath }) => (
                            <div
                                key={styleName}
                                className={`style-card ${
                                    selectedStyle === styleName ? "selected" : ""
                                }`}
                                onClick={() => handleStyleSelect(styleName)}
                            >
                                <LazyLoadImage
                                    src={`http://localhost:5001${imagePath}`}
                                    alt={styleName}
                                    effect="blur"
                                    onError={(e) =>
                                        console.error(`Error loading thumbnail for ${styleName}:`, e)
                                    }
                                />
                                <p>{styleName}</p>
                            </div>
                        ))}
                    </div>
                </div>

                <button
                    className="apply-btn"
                    onClick={handleApplyStyle}
                    disabled={isLoading}
                >
                    {isLoading ? "Applying Style..." : "Apply Style"}
                </button>
            </div>
        </div>
    );
};

export default App;
