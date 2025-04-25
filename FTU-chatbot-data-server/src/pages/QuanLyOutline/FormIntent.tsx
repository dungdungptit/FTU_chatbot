/* eslint-disable no-underscore-dangle */
import { IIntentRecord } from '@/models/topic';
import { ip } from '@/utils/ip';
import rules from '@/utils/rules';
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
} from 'antd';
import React, { useEffect, useState } from 'react';
import { useModel } from 'umi';

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
  const topicModel = useModel('outline');
  const questionModel = useModel('question');

  const handleFinish = async (values: IIntentRecord) => {
    console.log(topicModel.edit, 'values');

    if (topicModel.edit) {
      if (topicModel?.record?.id) {
        const payload = {
          data: {
            name: values.name,
            description: values.description,
          }
        }
        await topicModel.upd({ id: topicModel?.record?.id, data: payload });
        topicModel.setVisibleForm(false);
        topicModel.setEdit(false);
        topicModel.getData();
        // topicModel.getDataById(topicModel?.record?.id ?? '');
      } else {
        const payload = {
          data: {
            name: values.name,
            description: values.description,
          }
        }
        await topicModel.add(payload);
        topicModel.setVisibleForm(false);
        topicModel.setEdit(false);
      }
    } else {
      const payload = {
        data: {
          name: values.name,
          description: values.description,
        }
      }
      await topicModel.add(payload);
      topicModel.setVisibleForm(false);
      topicModel.setEdit(false);
    }
  };

  const handleFinishFile = async (values: any) => {
    const data = document.getElementById('formQuestionFile') as HTMLFormElement;
    const formData = new FormData(data);
    // formData.append('topic_id', IntentId);
    await questionModel.addFile(formData);
    topicModel.setVisibleForm(false);
    topicModel.setEdit(false);
    topicModel.getData();
  };
  console.log(form.getFieldsValue(), 'form.getFieldValue()');
  return (
    <Spin spinning={topicModel.loading}>
      {topicModel.edit ? (
        <Card title={'Chỉnh sửa'}>
          <Form
            id="formIntent"
            layout='vertical'
            form={form}
            onFinish={handleFinish}
            initialValues={{
              ...(topicModel?.record ?? {}),
            }}
          >
            {/* <Form.Item label="Mã chủ đề" name="name">
            <Input placeholder="Mã chủ đề" name="name" />
          </Form.Item> */}
            <Form.Item label="Tên chủ đề" name="name">
              <Input placeholder="Tên chủ đề" name="name" />
            </Form.Item>
            <Form.Item label="Mô tả chủ đề" name="description">
              <Input placeholder="Mô tả chủ đề" name="description" />
            </Form.Item>
            <Form.Item {...tailLayout}>
              <Button type="primary" htmlType="submit">
                {topicModel.edit ? 'Cập nhật' : 'Thêm mới'}
              </Button>
            </Form.Item>
          </Form>
        </Card>
      ) : (
        <Card title="Thêm mới">
          <Tabs>
            <TabPane tab="Nhập dữ liệu" key="tab-a">
              <Form
                id="formIntent"
                form={form}
                layout='vertical'
                onFinish={handleFinish}
                initialValues={{
                  ...(topicModel?.record ?? {}),
                }}
              >
                {/* <Form.Item label="Mã chủ đề" name="name">
                <Input placeholder="Mã chủ đề" name="name" />
              </Form.Item> */}
                <Form.Item label="Tên chủ đề" name="name">
                  <Input placeholder="Tên chủ đề" name="name" />
                </Form.Item>
                <Form.Item label="Mô tả chủ đề" name="description">
                  <Input placeholder="Mô tả chủ đề" name="description" />
                </Form.Item>
                <Form.Item {...tailLayout}>
                  <Button type="primary" htmlType="submit">
                    {'Thêm mới'}
                  </Button>
                </Form.Item>
              </Form>
            </TabPane>
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
          </Tabs>
        </Card>
      )}
    </Spin>
  );
};

export default FormIntent;
