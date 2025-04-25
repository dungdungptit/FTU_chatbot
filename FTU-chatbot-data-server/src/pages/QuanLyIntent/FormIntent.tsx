/* eslint-disable no-underscore-dangle */
import { IIntentRecord } from '@/models/intent';
import topic from '@/services/Outline/outline';
import { ip } from '@/utils/ip';
import rules from '@/utils/rules';
import { DeleteOutlined } from '@ant-design/icons';
import {
  Form,
  Input,
  Button,
  Card,
  Spin,
  DatePicker,
  Tabs,
  InputNumber,
  Col,
  Row,
  Select,
  Divider,
  Popconfirm,
} from 'antd';
import TextArea from 'antd/lib/input/TextArea';
import React, { useEffect, useState } from 'react';
import { history, useModel } from 'umi';

const layout = {
  labelCol: { span: 6 },
  wrapperCol: { span: 16 },
};
const tailLayout = {
  wrapperCol: { offset: 8, span: 16 },
};

const { TabPane } = Tabs;

const FormIntent = () => {
  const [form] = Form.useForm();
  const [status, setStatus] = useState<boolean>(false);
  const intentModel = useModel('intent');
  const questionModel = useModel('question');
  const pathname = window.location.pathname;
  const topicID = pathname.split('/')[2];
  const IntentID = pathname.split('/')[4];
  console.log(topicID, 'topicID');
  const handleFinish = async (values: IIntentRecord) => {
    if (intentModel.edit) {
      if (intentModel?.record?.id) {
        const payload = {
          data: {
            name: values.name,
            description: values.description,
            topic: {
              id: topicID,
            },
            file: values.file,
          }
        }
        await intentModel.upd({ id: intentModel?.record?.id, data: payload, topicID: topicID });
        intentModel.setVisibleForm(false);
        intentModel.setEdit(false);
        if (topicID) intentModel.getData(topicID)
        if (IntentID) intentModel.getDataById(IntentID)
        // intentModel.getDataById(intentModel?.record?.id ?? '');
      } else {
        const payload = {
          data: {
            name: values.name,
            description: values.description,
            topic: {
              id: topicID,
            },
            file: values.file,
          }
        }
        await intentModel.add(payload);
        if (topicID) intentModel.getData(topicID)
        if (IntentID) intentModel.getDataById(IntentID)
      }
    } else {
      const payload = {
        data: {
          name: values.name,
          description: values.description,
          topic: {
            id: topicID,
          },
          file: values.file,
        }
      }
      await intentModel.add(payload);
      intentModel.setVisibleForm(false);
      intentModel.setEdit(false);
      if (topicID) intentModel.getData(topicID)
      if (IntentID) intentModel.getDataById(IntentID)
    }
  };

  const handleDel = async (record: IIntentRecord) => {
    console.log("record delete", record);

    await intentModel.del(record?.id ?? "").then(() => {
      intentModel.setVisibleForm(false);
      intentModel.setEdit(false);
      if (topicID) intentModel.getData(topicID)
      history.push(`/topic/${topicID}/intent`);
    });
  };

  const handleFinishFile = async (values: any) => {
    const data = document.getElementById('formQuestionFile') as HTMLFormElement;
    const formData = new FormData(data);
    // formData.append('intent_id', IntentId);
    await questionModel.addFile(formData);
    intentModel.setVisibleForm(false);
    intentModel.setEdit(false);
    intentModel.getData(topicID);
  };
  console.log(form.getFieldsValue(), 'form.getFieldValue()');
  return (
    <Spin spinning={intentModel.loading}>
      {intentModel.edit ? (
        <Card title={'Chỉnh sửa'}>
          <Form
            id="formIntent"
            layout='vertical'
            form={form}
            onFinish={handleFinish}
            initialValues={{
              ...(intentModel?.record ?? {}),
            }}
          >
            {/* <Form.Item label="Mã chủ đề" name="intent_name">
            <Input placeholder="Mã chủ đề" name="intent_name" />
          </Form.Item> */}
            <Form.Item label="Tên thông tin" name="name">
              <Input placeholder="Tên thông tin" name="name" />
            </Form.Item>
            <Form.Item label="Mô tả thông tin" name="description">
              <Input placeholder="Mô tả thông tin" name="description" />
            </Form.Item>
            <Form.Item label="Nội dung thông tin" name="file">
              <TextArea placeholder="Nội dung thông tin" style={{
                minHeight: '500px'
              }} />
            </Form.Item>
            <Form.Item {...tailLayout}>
              <Button type="primary" htmlType="submit" style={{ marginRight: '8px' }}>
                {intentModel.edit ? 'Cập nhật' : 'Thêm mới'}
              </Button>
              <Popconfirm
                title="Bạn có muốn xóa?"
                okText="Có"
                cancelText="Không"
                onConfirm={() => handleDel(intentModel?.record)}
              >
                <Button type="primary">Xóa</Button>
              </Popconfirm>
            </Form.Item>
          </Form>
        </Card>
      ) : (
        <Card title="Thêm mới">
          {/* <Tabs>
            <TabPane tab="Nhập dữ liệu" key="tab-a"> */}
          <Form
            id="formIntent"
            form={form}
            layout='vertical'
            onFinish={handleFinish}
            initialValues={{
              ...(intentModel?.record ?? {}),
            }}
          >
            {/* <Form.Item label="Mã chủ đề" name="intent_name">
                <Input placeholder="Mã chủ đề" name="intent_name" />
              </Form.Item> */}
            <Form.Item label="Tên thông tin" name="name">
              <Input placeholder="Tên thông tin" name="name" />
            </Form.Item>
            <Form.Item label="Mô tả thông tin" name="description">
              <Input placeholder="Mô tả thông tin" name="description" />
            </Form.Item>
            <Form.Item label="Nội dung thông tin" name="file">
              <TextArea placeholder="Nội dung thông tin" name="file" style={{
                minHeight: '500px'
              }} />
            </Form.Item>
            <Form.Item {...tailLayout}>
              <Button type="primary" htmlType="submit">
                {'Thêm mới'}
              </Button>
            </Form.Item>
          </Form>
          {/* </TabPane> */}
          {/* <TabPane tab="Nhập bằng file" key="tab-b">
              <Form {...layout} form={form} onFinish={handleFinishFile} id="formQuestionFile">
                <Form.Item label="File" name="file">
                  <Input type="file" name="file" />
                </Form.Item>
                <Form.Item {...tailLayout}>
                  <Button type="primary" htmlType="submit">
                    Thêm mới
                  </Button>
                </Form.Item>
              </Form>
              <Button type="primary">
                <a
                  download={'tep_tin_mau_chu_de.xlsx'}
                  href={`${ip}/media/format/question_post/tep_tin_mau_chu_de.xlsx`}
                >
                  Tải file mẫu
                </a>
              </Button>
              <br />
              <Card title="Mẫu file">
                <Row>
                  <Col span={12}>
                    <strong>Câu hỏi</strong>
                  </Col>
                  <Col span={12}>
                    <strong>Chủ đề</strong>
                  </Col>
                </Row>
                <Divider />
                <Row>
                  <Col span={12}>
                    <p>Câu hỏi thuộc chủ đề(có thể để trống)</p>
                  </Col>
                  <Col span={12}>
                    <p>Tên chủ đề</p>
                  </Col>
                </Row>
              </Card>
            </TabPane> */}
          {/* </Tabs> */}
        </Card>
      )
      }
    </Spin >
  );
};

export default FormIntent;
